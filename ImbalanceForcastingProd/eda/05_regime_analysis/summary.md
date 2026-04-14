# Regime Analysis: Synthesis

This document synthesizes findings from all EDA sections to answer the key strategic questions about the spread trading model's environment.

## Is the spread compression seasonal or structural?

**Both, but primarily structural.**

Evidence for structural compression:
- Mean |spread| has declined 61% from 108 (2024) to 42 (2026) EUR/MWh
- The IDM-settlement R-squared has increased from 0.14 to 0.24, indicating market maturation
- The negative settlement price frequency dropped from 42.6% (2024) to 3.9% (2026)
- The year-over-year overlay shows 2026 running below 2025 at matching calendar days
- Convergence paths have flattened: later OB snapshots provide diminishing marginal information

Evidence for seasonal component:
- Winter months consistently show wider spreads than summer
- The hour-of-day pattern (peak at 17-21) is stable across years
- Both effects compound: summer 2026 will likely see the narrowest spreads yet

## Which features lost predictive power?

From the temporal stability analysis (04):
- **da_price**: importance dropped from 0.098 (2025) to 0.064 (2026) -- the main driver
- **da_demand**: 0.049 -> 0.024
- **reg_rmean8**: 0.040 -> 0.022
- **proxy_dev_from_hour**: 0.025 -> 0.015
- **temp_rmean24h**: 0.044 -> 0.032

Features **gaining** importance in 2026:
- **prod_momentum**: 0.000 -> 0.061 (newly relevant -- production data became available)
- **hour_cos**: 0.028 -> 0.059 (time-of-day matters more as spread narrows)
- **xborder_momentum**: 0.000 -> 0.026

The overall correlations between individual features and the spread target are weak (all |r| < 0.07). This is expected -- the model's value comes from combining many weak signals, not from any single strong predictor. The nonlinear binned-mean patterns in 03 show that LightGBM can extract signal that linear correlation misses.

## What is the current trading environment?

### Favorable factors:
- **Spread still exists**: mean |spread| of 42 EUR/MWh in 2026 remains substantial
- **Direction accuracy holds**: ~76-80% monthly accuracy in 2026 vs ~80-85% in 2025
- **Improving liquidity**: BA spreads have tightened, reducing execution costs
- **Model adapts**: walk-forward monthly P&L remains positive in all 2026 months tested
- **Production/cross-border data**: newly available features are adding signal

### Challenging factors:
- **Spread compression**: the opportunity set is shrinking
- **Lower tail frequency**: frac |spread|>10 dropped from 90% to 75%
- **Weaker autocorrelation**: lag-1 ACF of 0.217 limits persistence-based strategies
- **Feature-label correlations are universally weak** (all |r| < 0.07)

## Outlook

The spread compression trend is likely to continue as:
1. Market participants continue learning the 15-min settlement system
2. IDM liquidity improves, making OB prices better predictors of settlement
3. More sophisticated trading algorithms enter the market

However, the model still generates positive P&L because:
1. Settlement price extremes (driven by physical imbalance) will persist
2. The 2h lead time gives a structural information advantage
3. The model captures nonlinear hour-of-day and weather interactions that linear traders miss

**Recommendation**: The trading strategy should increasingly emphasize:
- **Selective high-conviction trades** over flat sizing (confidence scaling)
- **Dynamic threshold adjustment** as mean |spread| changes seasonally
- **Feature monitoring** for regime shifts (especially production and cross-border data)
- **Seasonal position sizing** (wider in winter, tighter in summer)
