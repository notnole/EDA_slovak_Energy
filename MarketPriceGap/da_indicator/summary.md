# Day-Ahead Trading Indicator for the Slovak Electricity Market

## Overview

A rule-based day-ahead indicator built from D-1 available information only (before the 11:00 DA auction gate closure). Tested on two use cases:

- **Speculative trader**: DA-IDM spread trading (1 MWh per trade)
- **Physical trader**: 1 MW / 2 MWh battery energy storage system (BESS)

**Data period**: Jan 2025 - Feb 2026 (speculative, IDM data required), Jan 2024 - Feb 2026 (BESS, DA-only).

---

## Headline Results

| Metric | Value |
|--------|-------|
| **Speculative total P&L** | 70,161 EUR |
| **Speculative accuracy** | 55.1% (traded hours) |
| **Speculative Sharpe** | 15.2 (annualized) |
| **BESS total revenue** | 177,439 EUR |
| **BESS avg capture rate** | 85.0% of perfect foresight |
| **BESS vs naive** | 2.3x better than fixed-schedule |

### 2026 Signal Persistence

| Metric | 2025 | 2026 (34 days) | Status |
|--------|------|-----------------|--------|
| Speculative accuracy | 54.3% | **64.8%** | IMPROVED |
| Speculative avg P&L/trade | +6.35 EUR | +**35.93 EUR** | IMPROVED |
| BESS capture rate | 82.0% | 78.7% | OK (slight decline) |
| BESS avg daily revenue | 226 EUR | 75 EUR | Lower (compressed spreads) |

**Conclusion**: The speculative signal is NOT disappearing in 2026 - it has actually strengthened. The BESS capture rate is slightly lower but remains well above the naive benchmark (which went deeply negative from Oct 2025).

---

## Signal Rules: What Works and What Doesn't

The speculative indicator uses 5 hierarchical rules. A deeper rule breakdown reveals stark differences:

### Rule Performance Summary

| Rule | Triggers | Accuracy | Avg P&L | Total P&L | Verdict |
|------|----------|----------|---------|-----------|---------|
| R1: Weekend | Is weekend | 59.6% | +11.11 | +29,795 | **Core signal** |
| R2: High Load Dev | Load forecast > normal + 85 MW | 50.7% | +5.80 | +24,601 | Profitable via magnitude |
| R3: Large Persistence | \|Yesterday spread\| >= 20 EUR | **68.4%** | +22.55 | +17,388 | **Best edge** |
| R4: Moderate Persistence | \|Yesterday daily spread\| >= 10 EUR | **47.7%** | **-3.02** | **-1,623** | **ELIMINATE** |
| R5: Baseline (no-trade) | Default | 56.8%* | +2.06* | +2,298* | Correct to skip |

*R5 hypothetical (not traded in backtest)

### Rule Stability: 2025 vs 2026

| Rule | 2025 Acc | 2026 Acc | Trend |
|------|----------|----------|-------|
| R1: Weekend | 59.2% | **65.1%** | Improving |
| R2: High Load Dev | 49.5% | **66.0%** | Dramatic improvement |
| R3: Large Persistence | 68.5% | 67.2% | Stable |
| R4: Moderate Persistence | 48.4% | **21.4%** | Collapsing - remove |

### Key Insights

1. **R3 (Large Persistence) is the highest-quality signal**: 68% accuracy in both years. When yesterday's same-hour spread was >= 20 EUR, following it works consistently. This is a genuine structural feature of the Slovak market - large pricing dislocations persist.

2. **R1 (Weekend) is the volume workhorse**: DA systematically overprices vs IDM on weekends (59.6% accuracy). This reflects lower weekend liquidity and conservative DA bidding. The effect strengthened in winter 2026 (65.1%).

3. **R2 (High Load Deviation) profits from magnitude, not accuracy**: At 50.7% accuracy it's essentially a coin flip, but when it's right the spreads are large (avg +36 EUR in 2026). This rule benefits from the correlation between high load and volatile prices. The dramatic 2026 improvement (49.5% -> 66.0%) may be sample-size noise (n=312) or may reflect the winter 2026 cold snap making load forecasts more systematically biased.

4. **R4 (Moderate Persistence) is a losing rule**: 47.7% accuracy, negative P&L in both years, and catastrophically bad in 2026 (21.4%). This should be removed entirely. The moderate daily spread threshold (10 EUR) is too noisy to be predictive.

5. **R5 (Baseline/no-trade) was a correct decision**: These hours would have been 56.8% accurate but with tiny edge (+2.06 EUR avg). Filtering them out reduces risk without sacrificing much P&L.

### Recommended Improvement

Remove R4 (Moderate Persistence). This would have saved 1,623 EUR and eliminated a losing strategy. The revised rule cascade would be:
1. Weekend -> sell DA (+1)
2. High load deviation -> buy DA (-1)
3. Large same-hour persistence (|spread| >= 20) -> follow
4. Otherwise -> no trade

---

## BESS Strategy Details

The BESS strategy uses a weighted ensemble of three D-1 available price shape predictors:

| Predictor | Weight | Rationale |
|-----------|--------|-----------|
| Load forecast rank | 40% | Higher load = higher price (structural) |
| Yesterday same-hour DA price rank | 35% | Price shape persists day-to-day |
| 7-day-ago DA price rank | 25% | Weekly cycle capture |

### Hour Selection Quality

| Metric | Result |
|--------|--------|
| Discharge hour overlap with actual top-2 | **52.8%** |
| Charge hour overlap with actual bottom-2 | **21.9%** |
| Most common discharge | h19 (53%), h20 (41%), h18 (25%) |
| Most common charge | h3 (37%), h2 (16%), h23 (16%) |

The indicator excels at identifying expensive hours (evening peak h17-20) but is weaker at identifying the cheapest hours, which shift more unpredictably (sometimes midday solar surplus, sometimes deep night).

### BESS Revenue Comparison (2 cycles/day, persistent SoC)

| Strategy | Total Revenue | Daily Avg | Capture Rate |
|----------|--------------|-----------|-------------|
| Perfect foresight | 246,481 EUR | 323 EUR | 100% |
| **Indicator** | **192,322 EUR** | **252 EUR** | **78%** |
| Naive (fixed hours) | 102,109 EUR | 134 EUR | -- |

Year-over-year: 2024 76% -> 2025 79% -> 2026 (Jan) 80% capture rate. Simulation uses persistent SoC across days with terminal penalty (20 EUR) to prevent drift. See `scripts/bess_simulation.py`.

The naive strategy collapsed from Oct 2025 onward because the fixed-hour assumption broke down. The indicator's adaptive approach avoided this entirely.

### Capture Rate Distribution (2 cycles, persistent SoC)

- Overall: 78% of perfect foresight
- Most days: 60-100% range
- Year-over-year improving (76% -> 79% -> 80%)

---

## Information Timing

All features respect the D-1 timing constraint (available before 11:00 gate closure):

| Feature | Source | Available When |
|---------|--------|---------------|
| Calendar (weekend, month, day-of-week) | Deterministic | Always |
| Yesterday's DA prices by hour | OKTE | D-1 morning |
| Yesterday's IDM VWAP by hour | OKTE | D-1 morning |
| Yesterday's spreads | Computed | D-1 morning |
| 7-day rolling DA/IDM averages | Computed | D-1 morning |
| D load forecast | DAMAS/SEPS | D-1 (published day before) |
| 7-day rolling actual load | EDA/DAMAS | D-1 morning |
| Load forecast deviation vs normal | Computed | D-1 morning |
| Yesterday nuclear output | ProductionPerType | D-1 (ENTSO-E, ~2h lag) |

**Not used** (not available D-1): imbalance settlement data (D+1), real-time 3-min SCADA (live but doesn't help D-1 decisions). DA price forecasts (r=0.80, MAE=18 EUR/MWh after data correction) could be a useful addition -- see `da_forecast_analysis/`.

---

## Monitoring Recommendations

### Signals to Watch

1. **Weekend effect**: If DA-IDM weekend spread drops below +10 EUR for 3+ consecutive weekends, investigate IDM liquidity changes.

2. **Spread persistence decay**: Monitor R3 accuracy monthly. If it drops below 60% over a 3-month window, the market may be becoming more efficient.

3. **BESS capture rate**: Track 30-day rolling capture rate. Below 70% warrants investigation. Below 50% suggests the price shape model needs recalibration.

4. **Naive strategy divergence**: If the naive fixed-schedule BESS starts outperforming the indicator for >2 months, the price shape may have reverted to its historical pattern.

### Structural Risks

- **IDM market reform**: Any change to IDM auction structure could break spread persistence.
- **BESS proliferation**: More batteries in Slovakia would arbitrage away the price shape predictability.
- **Nuclear outage patterns**: If EMO scheduling changes, the load/nuclear features may need recalibration.
- **Cross-border coupling changes**: CZ-SK-HU-PL flow changes affect DA price formation.

---

## Files

| File | Description |
|------|-------------|
| `scripts/build_da_indicator.py` | Main indicator: data loading, features, signals, backtest, 9 charts |
| `scripts/rule_breakdown.py` | Deeper analysis: rule-by-rule performance, BESS hour selection, 2 charts |
| `data/indicator_signals.csv` | Hourly signals with all features |
| `data/speculative_daily.csv` | Daily speculative P&L |
| `data/speculative_monthly.csv` | Monthly speculative aggregation |
| `data/bess_persistent_daily.csv` | Daily BESS revenue (2 cycles, persistent SoC) |
| `data/bess_persistent_monthly.csv` | Monthly BESS revenue comparison |
| `data/bess_soc_timeseries.csv` | Full SoC trace across all days |
| `plots/01-09_*.png` | Main analysis charts |
| `plots/10_rule_breakdown.png` | Rule accuracy, avg P&L, total P&L comparison |
| `plots/11_rule_by_year.png` | Rule accuracy 2025 vs 2026 stability check |

---

## Charts Index

1. **01_speculative_cumulative_pnl** - Cumulative P&L over time with 2026 shading
2. **02_speculative_monthly_pnl** - Monthly P&L bars with accuracy annotations
3. **03_speculative_hourly** - Accuracy and avg P&L by hour of day
4. **04_rolling_accuracy** - 30-day rolling accuracy (seasonal cycle visible)
5. **05_bess_cumulative_revenue** - Indicator vs perfect foresight vs naive
6. **06_bess_monthly_revenue** - Monthly comparison of three strategies
7. **07_bess_capture_rate** - Histogram of daily capture rates
8. **08_year_comparison** - 2025 vs 2026 for both traders
9. **09_weekend_vs_weekday** - Weekend effect decomposition
10. **10_rule_breakdown** - Which rules drive performance (accuracy, avg P&L, total)
11. **11_rule_by_year** - Rule accuracy stability 2025 vs 2026
