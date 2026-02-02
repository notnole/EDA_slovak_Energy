# Residual Analysis

## Purpose
Analyze Stage 1 and Stage 2 residuals to find remaining exploitable signal.

## Key Findings

### Stage 1 Residuals
- Strong autocorrelation at lag 1h: r = 0.57
- This motivated the two-stage model approach

### Stage 2 Residuals (After Correction)
- Autocorrelation reduced to r = 0.19 (67% reduction)
- Near Gaussian distribution (skewness=0.1, kurtosis=1.8)
- No significant remaining patterns

### Stage 3 Test
- Attempted to predict Stage 2 residuals
- Gain: +0.2% (negligible)
- Autocorrelation after Stage 3: 0.025 (essentially noise)

## Conclusion
Stage 2 extracts most of the exploitable signal. Stage 3 adds nothing meaningful.

## Files
- `residual_deep_analysis.py` - Deep residual analysis
- `residual_analysis.py` - Initial residual exploration
- `residual_analysis.png` - Visualization
