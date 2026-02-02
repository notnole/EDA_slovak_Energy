# Historical Import Deviation Analysis

## Data Summary
- **Settlement periods analyzed**: 9,894
- **Date range**: 2025-10-08 to 2026-01-24
- **Days of data**: 108

---

## Key Finding 1: Correlation with Imbalance

**Import Deviation correlates with System Imbalance:**
- Pearson r = **0.736**
- Statistically significant (p < 0.001)

---

## Key Finding 2: Sign Prediction Accuracy

| Predictor | Overall Accuracy |
|-----------|------------------|
| Import Deviation | **75.7%** |
| Persistence | 79.6% |

---

## Key Finding 3: Sign Flip Prediction

Sign flips occur in **20.4%** of periods (2,016 flips).

| Predictor | Accuracy on Flips |
|-----------|-------------------|
| Import Deviation | **64.1%** |
| Persistence | 0.0% (by definition) |

**Import deviation significantly outperforms persistence when sign flips occur.**

---

## Key Finding 4: Conditional Strategy

Best threshold: **2 MWh**

Strategy:
```
IF |previous imbalance| < 2 MWh:
    USE Import Deviation sign
ELSE:
    USE Persistence (previous sign)
```

| Strategy | Accuracy |
|----------|----------|
| Conditional | **81.8%** |
| Import Deviation alone | 75.7% |
| Persistence alone | 79.6% |

---

## Comparison: Historical vs Production

| Metric | Production (127 periods) | Historical (9,894 periods) |
|--------|--------------------------|----------------------------|
| Sign flip rate | ~29% | 20.4% |
| Import Dev correlation | +0.713 | 0.736 |
| Import Dev sign acc (flips) | 62.2% | 64.1% |

---

## Conclusions

1. **Import deviation is a valid signal** - correlation confirmed over 9,894 periods
2. **Sign flip prediction** - import deviation helps predict regime changes
3. **Conditional strategy** - use import_dev for small imbalances, persistence for large

---

*Analysis date: 2026-01-31*
*Data: 9,894 settlement periods from historical cross-border data*
