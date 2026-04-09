# Literature Review: BESS Optimization & Spread Trading

## Context

This review covers state-of-the-art approaches for two strategies on the Slovak electricity market (OKTE):
1. **BESS scheduling** (1 MW / 2 MWh) across DA, IDM, and imbalance markets
2. **Speculative DA-IDM spread trading** with position sizing and risk management

Current performance baseline:
- BESS: 192,322 EUR (78% capture rate, rank-based DP scheduling, 2 cycles/day)
- Speculative: 70,161 EUR (rule-based DA-IDM spread, 3 signals, PF=1.40)

---

## 1. BESS Multi-Market Optimization

### 1.1 Current Approach vs Best Practice

The current BESS strategy uses a **deterministic rank prediction** (load forecast 40%, yesterday's rank 35%, weekly rank 25%) fed into a dynamic programming scheduler. This is a solid first approach but leaves significant value on the table.

**State-of-the-art approaches** (ranked by practical impact):

| Approach | Description | Expected Improvement | Complexity |
|----------|-------------|---------------------|------------|
| **Stochastic DP** | Model price uncertainty via scenarios | +5-15% capture rate | Medium |
| **Rolling-horizon MPC** | Re-optimize as new info arrives (IDM) | +10-20% with IDM | High |
| **Multi-market co-optimization** | Joint DA + IDM + imbalance bidding | +15-30% revenue | High |
| **ML price shape prediction** | Replace weighted ranks with LightGBM | +3-8% capture rate | Low-Medium |
| **Degradation-aware scheduling** | Penalize deep cycles properly | Reduces hidden costs | Low |

### 1.2 Stochastic Programming for BESS

The key insight from recent literature is that **risk-neutral BESS operators should skew toward intraday trading** where price uncertainty creates larger spreads, while **risk-averse operators should anchor on DA** where prices are more predictable.

**Multi-stage stochastic MILP** (Arxiv 2510.27528):
- Stage 1: DA market commitment (decided D-1)
- Stage 2: IDM adjustments (decided intraday)
- Stage 3: Imbalance settlement (realized)
- Risk quantified via CVaR at each stage
- Result: Risk-neutral traders gain 15-25% from IDM opportunities; risk-averse traders hedge effectively with DA

**Practical recommendation for our case:**
- Keep the DA rank-based DP as the primary schedule
- Add a **second optimization layer** for IDM position adjustments based on realized prices
- Use scenario-based uncertainty (e.g., 50 price scenarios from historical residuals)

### 1.3 Model Predictive Control (MPC)

MPC is the gold standard for real-time battery management:
- Predicts future states over a finite horizon (e.g., 6-24 hours)
- Re-optimizes every 15-60 minutes as new data arrives
- Naturally handles the DA->IDM->imbalance sequential decision process

**Key advantage**: MPC can exploit intraday price movements that a DA-only schedule misses. When IDM prices deviate significantly from DA, the MPC adjusts the charge/discharge schedule.

**For our 1 MW/2 MWh system:**
- DA schedule: Commit to charge/discharge hours via DP (current approach)
- IDM adjustment: Every 1h, re-run MPC with updated IDM prices and adjust volumes
- Imbalance exposure: Small residual positions settle at imbalance price (structural +20 EUR premium observed in our data)

### 1.4 Battery Degradation Modeling

**Critical finding**: Ignoring degradation can turn profitable trades into losses. A battery at 100% DoD degrades ~10x faster than at 10% DoD.

Key degradation factors for our system:
| Factor | Impact | How to Model |
|--------|--------|-------------|
| Cycle depth | Dominant factor, nonlinear | Rainflow counting + Woehler curve |
| C-rate | Higher rates = faster aging | Linear penalty per MW/h |
| Calendar aging | Background degradation | Fixed annual capacity fade (~2%/year) |
| Temperature | Accelerates all mechanisms | Assume controlled (indoor installation) |
| State of charge | High SoC accelerates calendar aging | Penalize time at SoC > 90% |

**Practical cost model:**
```
degradation_cost_per_cycle = battery_replacement_cost / (equivalent_full_cycles * (1 + depth_penalty))
```

For a 2 MWh LFP battery at ~200 EUR/kWh replacement cost:
- Replacement cost: ~400,000 EUR
- Expected cycles: ~5,000 at 80% DoD
- Cost per full cycle: ~80 EUR
- Cost per 50% cycle: ~30 EUR (non-linear, shallower is much cheaper)

**Current terminal_penalty=20.0 is reasonable** but should be replaced with an explicit degradation cost model in the objective function.

### 1.5 Value Stacking Across Markets

European BESS operators increasingly stack revenues across:
1. **DA energy arbitrage** (primary, ~40-60% of revenue)
2. **IDM trading** (adjustment, ~15-25%)
3. **Balancing/imbalance** (residual, ~10-20%)
4. **Frequency regulation (FCR/aFRR)** (~15-30%, if eligible)

For the Slovak market specifically:
- FCR/aFRR participation may not yet be available for BESS (regulatory check needed)
- The IDM-imbalance spread of +20 EUR/MWh found in our data suggests significant value in deliberate imbalance exposure
- **Key insight**: A small systematic long position on the imbalance market (buying from imbalance rather than IDM) may add 10-15% to total revenue

---

## 2. Speculative Spread Trading

### 2.1 Current Approach vs Best Practice

Current: 3 rule-based signals (hourly persistence PF=1.78, daily persistence PF=1.31, load deviation PF=1.10). Combined PF=1.40 on 1 MWh position.

**Improvement opportunities:**

| Approach | Expected Impact | Complexity |
|----------|----------------|------------|
| **ML signal generation** (LightGBM) | Replace rules with probability scores | Medium |
| **Ensemble of models** | Combine LightGBM, XGBoost, linear | Low |
| **Probabilistic forecasting** | Confidence-weighted position sizing | Medium |
| **Multi-spread portfolio** | Trade DA-IDM + DA-Imb + IDM-Imb | Medium |
| **Walk-forward validation** | More reliable backtest results | Low |
| **Kelly criterion sizing** | Optimal position sizing | Low |

### 2.2 Key Features for Spread Prediction

From literature and our data, the most important features for European electricity spread forecasting:

**Fundamental features (available D-1):**
1. Load forecast and deviation from rolling mean (currently used -- good)
2. Renewable generation forecasts (solar + wind) -- **missing from current model**
3. Cross-border scheduled flows (import/export balance)
4. DA price level itself (spread behavior differs at 50 vs 200 EUR/MWh)
5. Nuclear/thermal availability (maintenance schedules)
6. Temperature forecast (drives load surprise)

**Technical/persistence features:**
7. Yesterday's same-hour spread (currently used -- strongest signal)
8. Yesterday's daily average spread (currently used)
9. Rolling 7-day spread statistics (mean, std, skew)
10. Day-of-week encoding (weekend effect is strong)
11. Hour-of-day encoding

**Market microstructure features:**
12. IDM trading volume and VWAP trajectory
13. DA auction clearing price vs pre-auction expectations
14. Bid-ask spread on IDM (liquidity proxy)
15. Net imbalance volume history

**Feature importance from literature (JRC study):**
- Net Imbalance Volume: 28.6% importance
- Loss of Load Probability: 27.5%
- De-rated margins: 14.0%
- Month/seasonality: 8.9%

### 2.3 Model Architecture Recommendations

**Primary: LightGBM/XGBoost ensemble**
- LightGBM is consistently top-performing for electricity price forecasting
- XGBoost provides complementary predictions (different regularization)
- Stack with linear meta-learner (Lasso or Ridge)
- Target: probability of positive spread, not point forecast

**Secondary: Seasonal Attention BiLSTM (for extreme events)**
- 25-37% improvement in predicting extreme prices vs standard models
- Useful specifically for tail events where rule-based signals fail
- Higher complexity, only worthwhile if data volume supports it

**Recommended pipeline:**
```
Features -> LightGBM (spread direction probability)
         -> XGBoost (spread direction probability)
         -> Ridge meta-learner -> Final probability
         -> Kelly criterion -> Position size (0 to max MW)
```

### 2.4 Walk-Forward Validation Protocol

**Critical for credibility.** The current backtests may overfit to the sample period.

Recommended protocol:
1. **Training window**: 6-12 months rolling
2. **Validation gap**: 1 day (prevent leakage from lagged features)
3. **Test window**: 1 month (evaluate, then roll forward)
4. **Retraining frequency**: Monthly (re-fit model each month)
5. **Evaluation**: Cumulative P&L, rolling 30-day Sharpe, max drawdown

**Key metrics to track:**
- Profit Factor (PF) -- must remain > 1.2 out-of-sample
- Sharpe ratio -- target > 1.5 annualized
- Maximum drawdown (EUR and duration)
- Win rate by signal confidence bucket
- Hit rate degradation over time (regime change detection)

### 2.5 Risk Management Frameworks

**Kelly Criterion for position sizing:**
```
Kelly_fraction = (p * b - q) / b
where:
  p = estimated win probability
  b = average win / average loss ratio
  q = 1 - p
```

**Practical application:**
- Use **half-Kelly** (50% of optimal) to reduce variance
- For Signal 1 (PF=1.78, 68% accuracy): Full Kelly = 31%, Half Kelly = 15.5% of max position
- For Signal 3 (PF=1.10, ~55% accuracy): Full Kelly = 5%, Half Kelly = 2.5% -- barely worth trading

**CVaR (Conditional Value-at-Risk) framework:**
- Superior to VaR for electricity markets (fat tails, skewness)
- Set CVaR constraint at 95% level: maximum expected daily loss < X EUR
- Use for portfolio-level risk: combined BESS + speculative exposure
- Literature shows CVaR constraints reduce maximum drawdown by 30-50% with only 5-10% reduction in expected profit

**Stop-loss rules:**
- Per-signal: Suspend signal for 7 days if rolling 14-day PF drops below 0.8
- Portfolio: Reduce all positions by 50% if monthly drawdown exceeds 15% of trailing 3-month profit
- Emergency: Halt trading if 30-day rolling loss exceeds 2x average monthly profit

---

## 3. Multi-Spread Portfolio

### 3.1 Three Tradeable Spreads

Our data shows three distinct spreads with different characteristics:

| Spread | Avg (EUR/MWh) | Std | Persistence | Tradeable? |
|--------|---------------|-----|-------------|------------|
| DA - IDM | +8.7 | ~25 | High (r>0.5) | Yes (current) |
| DA - Imbalance | ~+28 | ~40 | Medium | Yes (high avg, volatile) |
| IDM - Imbalance | ~+20 | ~35 | Low | Marginal (structural) |

**Key insight from our data**: The IDM-Imbalance spread of +20 EUR/MWh is structural (IDM systematically prices above imbalance). This means:
- A **passive strategy** of always buying from imbalance rather than IDM has a structural edge
- The BESS should deliberately leave some volume for imbalance settlement when the indicator suggests it

### 3.2 Portfolio Diversification

Correlation between spread signals is likely moderate (~0.3-0.5). This means:
- A portfolio of 3 spread strategies has lower risk than any individual strategy
- Optimal allocation should weight by Sharpe ratio and correlation structure
- Use mean-CVaR optimization for allocation

---

## 4. Immediate Priorities for Maximum Improvement

Ranked by **expected impact / implementation effort**:

### Priority 1: Walk-Forward Validation Framework (LOW effort, CRITICAL)
- Implement proper expanding/rolling window backtests for all signals
- This doesn't improve performance but prevents false confidence
- Must be done before any strategy goes live

### Priority 2: ML-Based Spread Direction (MEDIUM effort, HIGH impact)
- Replace rule-based signals with LightGBM probability model
- Add renewable forecasts and cross-border flow features
- Expected improvement: PF from 1.40 to 1.6-2.0

### Priority 3: Kelly Criterion Position Sizing (LOW effort, MEDIUM impact)
- Scale position size by model confidence (probability)
- Expected improvement: 20-40% higher risk-adjusted returns with same capital

### Priority 4: BESS IDM Adjustment Layer (MEDIUM effort, HIGH impact)
- After DA commitment, adjust BESS schedule based on IDM prices
- Add deliberate imbalance exposure when IDM-Imbalance spread is wide
- Expected improvement: +15-25% total BESS revenue

### Priority 5: Explicit Degradation Model (LOW effort, MEDIUM impact)
- Replace terminal_penalty=20.0 with cycle-depth-aware cost
- Prevents uneconomic deep cycles
- Expected: minimal revenue change but more accurate profitability assessment

### Priority 6: Multi-Spread Portfolio (MEDIUM effort, MEDIUM impact)
- Trade DA-Imbalance spread alongside DA-IDM
- Use correlation structure for portfolio optimization
- Expected: 30-50% more trading opportunities

---

## 5. What NOT to Do

1. **Do not build a neural network for price prediction** -- insufficient data (2 years) for complex architectures. LightGBM/XGBoost is the right complexity level.
2. **Do not use absolute price forecasting for BESS** -- rank prediction is correct. Absolute price errors are large; rank errors are forgiving.
3. **Do not trade all hours equally** -- focus on hours with high spread variance (peaks, ramps). Many hours have near-zero expected spread.
4. **Do not ignore transaction costs** -- even small bid-ask spreads and slippage compound. Model explicitly.
5. **Do not backtest on the full sample and report results** -- walk-forward only.

---

## Sources

### BESS Optimization
- [Maximizing Battery Storage Profits via High-Frequency Intraday Trading](https://arxiv.org/html/2504.06932v3)
- [Risk-constrained stochastic scheduling of multi-market energy storage](https://arxiv.org/html/2510.27528v1)
- [Battery Storage Optimization: Value Stacking](https://flex-power.energy/energyblog/battery-storage-trading-strategy/)
- [Integrated energy scheduling with battery degradation-aware optimization](https://www.nature.com/articles/s41598-025-28469-6)
- [Optimal Battery Storage Participation in European Energy and Reserves Markets](https://www.mdpi.com/1996-1073/13/24/6629)
- [Robust market-based BESS management for European balancing markets](https://www.sciencedirect.com/science/article/pii/S2352152X24036685)
- [Battery degradation: Impact on economic dispatch](https://onlinelibrary.wiley.com/doi/10.1002/est2.588)
- [Degradation and cycling (Modo Energy)](https://modoenergy.com/research/battery-energy-storage-degradation-cycling-warranty)
- [Stochastic economic MPC for renewable energy and ancillary services](https://www.sciencedirect.com/science/article/abs/pii/S2352467724001024)

### Spread Trading & Price Forecasting
- [Probabilistic intraday electricity price forecasting using generative ML](https://arxiv.org/abs/2506.00044)
- [Deep Learning-Based Electricity Price Forecast for Virtual Bidding](https://arxiv.org/abs/2412.00062)
- [Seasonality in deep learning forecasts of electricity imbalance prices](https://www.sciencedirect.com/science/article/pii/S014098832400478X)
- [ML and Deep Learning Forecasts of Electricity Imbalance Prices](https://www.researchgate.net/publication/371748531)
- [Price Forecasting for the Balancing Energy Market Using ML (JRC)](https://publications.jrc.ec.europa.eu/repository/handle/JRC121309)
- [Hybrid stacking for short-term price forecasting](https://www.techrxiv.org/users/802208/articles/1235479)
- [EPEX ML spread trading (GitHub)](https://github.com/ekapope/EPEX-machine-learning)
- [Deep reinforcement learning for continuous intraday market bidding](https://link.springer.com/article/10.1007/s10994-021-06020-8)

### Risk Management
- [Mean-CVaR Optimal Energy Storage Operation (Princeton)](https://castle.princeton.edu/wp-content/uploads/2020/11/Moazeni-Mean-conditional-value-at-risk-optimal-energy-storage-operation-IEEE-July-20-2014.pdf)
- [CVaR-based planning model for integrated energy systems](https://arxiv.org/pdf/2104.10862)
- [Kelly Criterion: Risk-Constrained trading (QuantInsti)](https://blog.quantinsti.com/risk-constrained-kelly-criterion/)

### XBID / Intraday Market
- [Continuous Intraday Market Coupling Algorithm](https://link.springer.com/chapter/10.1007/978-3-031-86315-8_7)
- [ENTSOE SIDC overview](https://www.entsoe.eu/network_codes/cacm/implementation/sidc/)
- [Modeling Intraday Markets under XBID (German market evidence)](https://www.mdpi.com/1996-1073/12/22/4339)
