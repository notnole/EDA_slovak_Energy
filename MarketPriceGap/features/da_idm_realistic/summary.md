# DA-IDM Spread Direction Analysis - REALISTIC VERSION

## Critical Correction

**Previous analysis was WRONG** because it used same-day information (previous hour's spread).

**Reality**: DA position must be decided at D-1 (day before). You CANNOT use:
- Same-day spread direction
- Same-day prices
- Any lag < 24 hours

---

## Realistic Prediction Problem

**At DA auction time (D-1, ~noon), predict**: Will DA price be > or < IDM price tomorrow?

**Information available**:
- Yesterday's same hour outcome
- Historical patterns (last week, monthly averages)
- Calendar features (day of week, hour)
- Weather/load forecasts (not in this dataset)

---

## Key Findings

### Best Realistic Predictor: Yesterday Same Hour

| Metric | Value |
|--------|-------|
| Accuracy | **59.6%** |
| Win Rate | 59.5% |
| Total P&L | 92,784 EUR/MWh |
| Avg P&L per trade | 10.11 EUR/MWh |
| Sharpe (annualized) | 22.51 |

### Feature Importance (Correlation with Direction)

| Feature | Correlation | Interpretation |
|---------|-------------|----------------|
| spread_7d_same_hour_avg | 0.270 *** | Predictive |
| spread_yesterday_same_hour | 0.257 *** | Predictive |
| direction_yesterday_same_hour | 0.186 *** | Predictive |
| direction_lastweek_same_hour | 0.146 *** | Predictive |
| hour | -0.141 *** | Predictive |
| direction_2d_ago_same_hour | 0.119 *** | Predictive |

---

## Transition Probabilities (Key Insight)

The spread direction shows **day-to-day persistence** for the same hour:

| Yesterday State | P(Today DA>IDM) | P(Today DA<IDM) |
|-----------------|-----------------|-----------------|
| DA > IDM | ~63% | ~37% |
| DA < IDM | ~44% | ~56% |

**Interpretation**: If yesterday same hour had DA > IDM, there's ~63% chance today will too.

---

## Trading Strategy Rules

### Simple Rule (Follow Yesterday Same Hour)
```
For each delivery hour H on day D:
    1. Look at hour H on day D-1
    2. If DA > IDM yesterday: SELL on DA, buy back on IDM
    3. If DA < IDM yesterday: BUY on DA, sell on IDM
```

### Accuracy: 59.6%

This is **better than random (50%)** but **not as strong as the incorrect 78%** from same-day analysis.

---

## Realistic Performance Comparison

| Strategy | Accuracy | Description |
|----------|----------|-------------|
| 7d same hour avg positive -> sell | 59.6% | Best |
| Follow yesterday same hour | 59.6% |  |
| Follow last week same hour | 57.6% |  |
| Follow 2 days ago same hour | 56.3% |  |
| Naive: always sell DA | 54.1% |  |
| Yesterday avg positive -> sell | 50.2% |  |
| Naive: always buy DA | 45.9% |  |
| Weekday: sell, Weekend: buy | 48.4% |  |
| Peak hours sell DA | 44.7% |  |

---

## When to Trade (Conditional Analysis)

The strategy works better in certain conditions:

1. **Time of day**: Peak hours (8-20h) tend to be more predictable
2. **Large spreads yesterday**: When yesterday's spread was large, direction is more likely to persist
3. **Weekdays**: Slightly more predictable than weekends

---

## Limitations & Next Steps

### Current Limitations
- Only ~59.6% accuracy (vs 50% random)
- No weather/forecast features included
- Transaction costs not modeled

### To Improve Predictions
1. Add day-ahead weather forecasts
2. Add day-ahead load forecasts
3. Add scheduled outage information
4. Build ML model combining all D-1 features
5. Consider ensemble of yesterday + last week + seasonal patterns

---

## Files Generated

- `01_realistic_features.png` - Feature correlations and transition probabilities
- `02_realistic_rules.png` - Trading rule comparison
- `03_when_to_trade.png` - Conditional accuracy analysis
- `summary.md` - This summary
