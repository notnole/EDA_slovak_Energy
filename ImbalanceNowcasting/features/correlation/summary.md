# Feature-Label Correlation Analysis Summary

## Purpose
Analyze how features correlate with the imbalance label to identify the most predictive signals.

## Correlation Results

| Feature | Correlation (r) | R² | Rank |
|---------|----------------|-----|------|
| Regulation Deviation | -0.672 | 45.2% | 1 |
| Regulation | -0.667 | 44.5% | 2 |
| Load Deviation | -0.116 | 1.3% | 3 |
| Load | -0.101 | 1.0% | 4 |
| Production | -0.050 | 0.25% | 5 |
| Production Deviation | -0.042 | 0.18% | 6 |
| Export/Import | +0.024 | 0.06% | 7 |
| Export/Import Deviation | +0.006 | 0.00% | 8 |

## Key Findings

### Regulation is dominant
- Explains ~45% of imbalance variance alone
- This makes sense: regulation = -imbalance (by definition, roughly)
- Deviation slightly better than raw (+0.5%)

### Load adds marginal value
- Only 1% additional variance explained
- Deviation 15% better than raw (0.116 vs 0.101)
- Independent signal from regulation (correlation ~0.1)

### Production & Export/Import not useful
- Near-zero correlation with imbalance
- Limited data period may contribute
- Can be dropped from model

## Feature Inter-correlations
- Regulation vs Regulation_deviation: 0.97 (almost same)
- Load vs Load_deviation: 0.71 (deviation removes predictable part)
- Regulation vs Load: ~0.1 (independent signals)

## Model Implications

1. **Regulation is the key feature** - must include
2. **Use deviation features** - slightly better than raw
3. **Drop production/export_import** - no predictive power
4. **Remaining variance (~55%)** needs: time features, lag features, within-QH dynamics
5. **Target baseline**: With just regulation, expect R² ~ 0.45, RMSE ~ 9.4 MWh
