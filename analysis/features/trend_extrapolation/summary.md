# Load Trend Extrapolation Test Summary

## Purpose
Test if we can learn seasonal patterns from 2024 and apply them to 2025-2026 data to create a valid "deviation" feature.

This is critical because:
- In production, we only have historical patterns
- The feature must generalize to future data

## Test Setup
- **Train**: 2024 data (168K rows)
- **Test**: 2025-2026 data (180K rows)
- **Methods tested**:
  1. ToD Deviation (avg by hour+minute+weekend from 2024)
  2. STL Seasonal (daily pattern learned from 2024)

## Results

| Method | 2024 (train) | 2025-2026 (test) | Change |
|--------|-------------|------------------|--------|
| Raw Load | -0.099 | -0.110 | +11% |
| ToD Deviation | -0.109 | **-0.137** | +26% |
| STL Seasonal | - | **-0.155** | +41% |

## Key Findings

### 1. ToD Deviation extrapolates well
- Correlation IMPROVES from -0.109 (train) to -0.137 (test)
- 24% better than raw load on test data
- Safe to use as feature

### 2. STL seasonal-only is even better
- Correlation of -0.155 on test data
- 41% better than raw load
- But requires more complex computation

### 3. Why does it work better on test?
- The bottom-left plot shows: 2025 load is systematically HIGHER than 2024
- This offset creates larger deviations
- Larger deviations = stronger correlation signal

## Visual Observations

The bottom-left plot shows:
- **Orange line**: Expected load (from 2024 patterns)
- **Blue line**: Actual load in Jan 2025
- **Gap**: 2025 is ~200-400 MW higher than 2024 baseline

This offset is a "trend" that ToD deviation captures as deviation.

## Conclusion

**[VALIDATED]** ToD Deviation is a usable feature:
1. Extrapolates from 2024 to 2025-2026 without degradation
2. Actually improves correlation on out-of-sample data
3. Simple to compute (just lookup table by hour/minute/weekend)
4. No need for complex STL in production

## Model Implication

Use `load_deviation = load - avg_load_at_same_ToD` as a feature.
The lookup table can be precomputed from training data.
