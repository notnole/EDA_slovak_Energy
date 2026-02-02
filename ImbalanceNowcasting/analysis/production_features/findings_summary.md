# Production Feature Analysis - Key Findings

## Data
- **127 settlement periods** with full feature set
- **~1.5 days** of production data (Jan 29-31, 2026)

---

## Key Findings

### Finding 1: Disagreement is a Warning Signal

When Model and Import Deviation **AGREE** on sign:
- Error rate: **17.1%**
- Model accuracy: **82.9%**

When Model and Import Deviation **DISAGREE**:
- Error rate: **39.2%**
- Model accuracy: **60.8%**

**Implication**: If model and import deviation disagree, reduce confidence in prediction.

---

### Finding 2: Import Deviation Excels at Sign Flips

When the actual sign **FLIPS** from previous period:

| Predictor | Accuracy on Flips |
|-----------|-------------------|
| Import Deviation | **62.2%** |
| Frequency | 54.1% |
| Model | 32.4% |
| Persistence | 0.0% |

**Import deviation is TWICE as good as the model at predicting the new sign when a flip occurs!**

---

### Finding 3: Import Deviation Predicts Regulation Change

Correlation with regulation change (final period mean - first observation):
- **Import Deviation: r = +0.208** (significant, p < 0.05)
- Frequency: r = +0.022 (not significant)
- Generation: r = -0.030 (not significant)
- Load: r = -0.046 (not significant)

**Import deviation knows something about how regulation will evolve during the period.**

---

### Finding 4: Previous Imbalance Magnitude Predicts Flips

| Predictor | Correlation with Flip |
|-----------|----------------------|
| \|Previous imbalance\| | **r = -0.438** (significant) |
| \|Import deviation\| | r = -0.155 (not significant) |
| Frequency deviation | r = -0.125 (not significant) |

**When previous imbalance is small (< 3 MWh), sign flips are much more likely.**

---

## Proposed Strategy: Conditional Sign Prediction

```
IF |previous imbalance| < 3 MWh:
    USE Import Deviation sign  (better at flips: 70% vs 60%)
ELSE:
    USE Model sign             (better when stable: 83% vs 63%)
```

### Performance

| Strategy | Accuracy | 95% CI |
|----------|----------|--------|
| Model alone | 73.8% | [65.9%, 81.0%] |
| Conditional | **77.8%** | [70.6%, 84.9%] |

**Improvement: +4.0 percentage points**

### Caveat
- Confidence intervals overlap
- Need more data (~500+ observations) to confirm significance
- Currently based on only 127 settlement periods

---

## Recommendations

1. **Continue collecting production data** with all features (import_deviation, frequency)

2. **Implement disagreement warning**: When model and import_deviation disagree, flag prediction as low confidence

3. **Test conditional strategy** in paper trading:
   - When |prev| < 3: use import_deviation sign
   - Otherwise: use model sign

4. **Monitor flip prediction**: Track whether import deviation continues to outperform model on sign flips

5. **Re-analyze after 1 week** (~672 periods) for statistically significant conclusions

---

## Physical Interpretation

**Why might import deviation predict sign flips?**

Import deviation = Measured - Scheduled cross-border flow

When import_deviation > 0:
- More power is flowing INTO Slovakia than scheduled
- This extra power must be absorbed → positive imbalance likely
- If regulation is currently negative (undersupply), expect flip to positive

The import deviation represents an **external forcing** on the system that the grid operator may not have fully anticipated, leading to a sign flip in imbalance.

---

*Analysis date: January 31, 2026*
*Data: 127 settlement periods from production deployment*
