# Regulation Volatility & Daily Direction Analysis

## Part 1: Regulation Volatility Hypothesis

**Question**: Did the IDM-Imbalance arbitrage close because the grid became more stable?

**Theory**: High regulation volatility = system stress = unpredictable imbalance prices = larger spreads.

### Finding: HYPOTHESIS REJECTED

**Regulation volatility does NOT explain the arbitrage closure.**

| Metric | 2025 | Jan 2026 | Change |
|--------|------|----------|--------|
| Regulation Volatility | 27.5 MW | 25.3 MW | **-8.1%** |
| IDM-Imbalance Spread | +19.1 EUR | **-1.1 EUR** | **-106%** |
| Win Rate | 68% | **46%** | **-22 pp** |

The spread collapsed by over 100%, but regulation volatility barely changed (-8%).

---

## Correlation Analysis

### Overall
- Correlation (Reg Vol vs Spread): **0.040** (essentially zero)

### By Volatility Quartile

| Quartile | Avg Spread | Win Rate |
|----------|------------|----------|
| Q1 (Low) | +14.6 EUR | 67% |
| Q2 | +18.9 EUR | 68% |
| Q3 | +17.9 EUR | 66% |
| Q4 (High) | +20.1 EUR | 64% |

Higher volatility correlates weakly with higher spreads, but the effect is small (~5 EUR difference between Q1 and Q4).

---

## Monthly Pattern

| Month | Reg Vol (MW) | Spread (EUR) | Win Rate |
|-------|--------------|--------------|----------|
| Jan 2025 | 28.1 | +37.4 | 74% |
| Feb 2025 | 30.0 | +36.1 | 72% |
| Mar 2025 | 29.0 | +29.5 | 74% |
| Apr 2025 | 31.4 | +18.6 | 62% |
| May 2025 | 26.9 | +8.0 | 59% |
| Jun 2025 | 28.7 | +11.9 | 61% |
| Jul 2025 | 29.0 | +16.7 | 69% |
| Aug 2025 | 26.5 | +8.7 | 61% |
| Sep 2025 | 28.1 | +17.4 | 66% |
| Oct 2025 | 26.1 | +10.0 | 66% |
| Nov 2025 | 23.7 | +17.8 | 75% |
| Dec 2025 | 23.4 | +18.0 | 72% |
| **Jan 2026** | 25.3 | **-1.1** | **46%** |

Note: Jan 2025 and Jan 2026 had similar regulation volatility (28.1 vs 25.3 MW), but spreads differed by 38 EUR!

---

## Implications

### What This Means
1. **Grid stability is NOT the cause** - The system wasn't running more smoothly in Jan 2026
2. **Market efficiency increased** - IDM and Imbalance prices became more correlated through OTHER mechanisms
3. **Possible causes**:
   - More market participants leading to tighter spreads
   - Better forecasting by TSO/market participants
   - Regulatory changes affecting price formation
   - Increased cross-border balancing coordination

### Trading Implications
- Cannot use regulation volatility as a regime indicator
- Need to find other signals that predict when arbitrage will re-open
- The "always sell IDM" strategy is dead regardless of grid conditions

---

---

## Part 2: Daily Direction Predictability

**Key Insight**: The spread didn't disappear - it became **bidirectional**. The daily average is near zero because it's swinging positive AND negative, not because spreads are small.

### Direction Distribution

| Period | Positive Days | Negative Days | Avg Spread | Spread Std |
|--------|---------------|---------------|------------|------------|
| 2025 | **83%** | 17% | +19.0 EUR | 31.0 EUR |
| Jan 2026 | **46%** | 54% | -1.1 EUR | **25.5 EUR** |

**The spread volatility (25-31 EUR) is still substantial** - direction just flipped from predictable to random.

### Persistence Collapsed

| Condition | 2025 | Jan 2026 |
|-----------|------|----------|
| P(+ve today \| +ve yesterday) | **87%** | 50% |
| P(+ve today \| -ve yesterday) | 67% | 43% |
| Persistence Strength | **+20pp** | **+7pp** |

In 2025, yesterday strongly predicted today. In Jan 2026, it's nearly random.

### Day-of-Week Pattern Broke

| Day | 2025 % Positive | Jan 2026 % Positive |
|-----|-----------------|---------------------|
| Monday | 94% | 33% |
| Tuesday | 90% | **0%** |
| Wednesday | 81% | 33% |
| Thursday | 77% | 50% |
| Friday | 90% | 50% |
| Saturday | 75% | 50% |
| Sunday | 75% | **100%** |

The reliable Monday/Tuesday pattern completely inverted!

### Simple Prediction Rules (Jan 2026)

| Rule | Accuracy |
|------|----------|
| Always negative | 54% |
| Follow yesterday | 54% |
| Follow 3-day trend | 50% |
| Always positive | 46% |
| Weekday +ve, Weekend -ve | 33% |

**Best rule: 54%** - barely better than coin flip. Simple rules don't work.

### Trading P&L Simulation (Jan 2026, 1 MWh)

| Strategy | P&L |
|----------|-----|
| Always Sell IDM | **-447 EUR** |
| Follow Yesterday | **+1,029 EUR** |
| Perfect Foresight | **+8,116 EUR** |

The spread VALUE is still there (+8k EUR potential), but capturing it requires accurate direction prediction.

---

## Key Conclusions

1. **The arbitrage isn't dead - it's hidden**: ~25 EUR daily spread still exists
2. **Direction became unpredictable**: 83% → 46% positive days
3. **Persistence collapsed**: Yesterday no longer predicts today
4. **Simple rules don't work**: Best accuracy is 54%
5. **Opportunity remains**: Perfect foresight yields +8k EUR/month
6. **Need ML model**: To capture the spread, need features that predict daily direction

---

## Part 3: Predictive Features for Daily Direction

### Best Features (Jan 2026)

| Feature | Correlation | Accuracy |
|---------|-------------|----------|
| Morning spread direction | r=0.66 | **87.5%** |
| Morning >50% positive hours | r=0.74 | **83.3%** |
| Previous afternoon spread direction | r=0.41 | **70.8%** |
| Morning imbalance price (inverse) | r=-0.58 | - |

### Key Finding: **MORNING PREDICTS THE DAY**

If the spread is positive in the morning (hours 0-6), the day will be positive **87.5% of the time**.

This is actionable:
- Wait until ~6-7 AM
- Check if IDM > Imbalance on average so far
- If yes → Sell IDM for the rest of the day
- If no → Buy IDM for the rest of the day

### DA Prices Became Irrelevant

| Feature | 2025 Correlation | Jan 2026 Correlation |
|---------|------------------|----------------------|
| DA price mean | +0.21 | **-0.04** |
| DA price min | +0.25 | +0.08 |

DA prices strongly predicted direction in 2025, but have **zero predictive power in Jan 2026**.

### Visual Evidence

Bottom-right chart in `04_feature_analysis.png` shows Jan 2026 daily pattern:
- Green bars (day positive) almost always had positive morning spread
- Red bars (day negative) almost always had negative morning spread
- Only ~3 mismatches in 24 days

### Trading Strategy Implication

```
At 6:00 AM:
  IF mean(IDM - Imbalance, hours 0-5) > 0:
      SELL IDM all day (spread likely positive)
  ELSE:
      BUY IDM all day (spread likely negative)

Expected accuracy: ~85%
```

### Caveats
- Sample size is small (24 days in Jan 2026)
- Need to test on Feb 2026 data when available
- Morning hours have lower liquidity

### Next Steps
- Validate on Feb 2026 data
- Backtest intraday strategy with hourly signals
- Investigate why DA prices lost predictive power

---

## Visualizations

1. **01_regulation_spread_analysis.png** - Regulation volatility vs spread (hypothesis test)
2. **02_rolling_correlation.png** - Rolling 7-day correlation over time
3. **03_daily_direction_analysis.png** - Daily direction patterns and predictability
4. **04_feature_analysis.png** - Feature correlations and morning spread signal

---

## Data

- **Source**: 3MIN_REG.csv (1.3M records, Jan 2024 - Jan 2026)
- **Merged with**: IDM prices, Imbalance settlement prices, DA prices
- **Output**:
  - data/hourly_reg_spread.csv (9,246 hourly records)
  - data/daily_direction.csv (388 daily records)
  - data/daily_with_features.csv (388 daily records with predictive features)
