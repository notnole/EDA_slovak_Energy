# Feature Decomposition Analysis Summary

## Purpose
Decompose features into trend, seasonal, and residual components to:
1. Understand how much variance is predictable
2. Create deviation features that may correlate better with imbalance

## STL Decomposition Results

### Load
| Component | Variance % |
|-----------|-----------|
| Trend | 53.0% |
| Seasonal | 38.2% |
| **Residual** | **3.6%** |

**Interpretation**: Load is 96% predictable from trend+seasonal. The residual (surprise) is tiny but may be informative for imbalance prediction.

### Regulation
| Component | Variance % |
|-----------|-----------|
| Trend | 14.9% |
| Seasonal | 27.6% |
| **Residual** | **43.6%** |

**Interpretation**: Regulation is mostly unpredictable (44% residual). The trend component is too noisy to extrapolate reliably. Use raw regulation or ToD deviation instead.

## Time-of-Day Deviation

An alternative to STL - compute deviation from average value at same time of day (split by weekday/weekend).

| Feature | Deviation Std |
|---------|--------------|
| Load | 335 MW |
| Regulation | 47 MW |
| Production | 218 MW |
| Export/Import | 283 MW |

## Model Implications

1. **Load residual/deviation is promising** - captures the "surprise" in load
2. **Regulation trend is NOT usable** - too random to extrapolate
3. **ToD deviation is simpler and robust** - doesn't require fitting a model
4. **Need to test**: Can we extrapolate 2024 load trend to 2025-2026?

## Next Steps
- Test trend extrapolation for load (train on 2024, test on 2025-2026)
- If residual stays correlated with imbalance, it's a valid feature
