# AR Correction & Bias Calibration Experiment

## Objective
Investigate whether post-hoc corrections can improve LightGBM predictions by exploiting systematic error patterns.

## Key Finding
**Hour x QH bias correction provides +1.7% average MAE improvement** when the model is trained WITHOUT the `qh_position` feature.

## Experiments Conducted

### 1. Proxy Quality Test
- Proxy formula: `imbalance ≈ -0.25 * mean(regulation_mw)`
- Correlation with actual: **0.926**
- Proxy MAE: 2.16 MWh
- Proxy error autocorrelation (lag-1): 0.398

### 2. AR Correction with Proxy Error
- Used lagged proxy error to correct predictions
- Result: Only **+1.2% average improvement**
- Conclusion: Model already captures most autocorrelation signal

### 3. Lead-0 as Truth Correction
- Used Lead-0 prediction as pseudo-truth for earlier leads
- Result: +0.3% (Lead 12), +2.9% (Lead 9)
- Limited practical value

### 4. Error Oscillation Analysis
Found systematic bias by quarter-hour position within hour:
- QH 1 & 3: Model over-predicts (negative error)
- QH 2 & 4: Model under-predicts (positive error)

This pattern is **extremely stable** (0.97 correlation between calibration and test).

### 5. Bias Calibration Experiment (Final)
Train model WITHOUT `qh_position`, learn bias patterns OOF, apply correction.

**Pattern Stability (calibration vs test correlation):**
| Pattern   | Avg Correlation | Stable? |
|-----------|-----------------|---------|
| QH        | 0.97            | YES     |
| Hour      | 0.65            | Moderate|
| DoW       | 0.50            | NO      |

**Improvement by Correction Type:**
| Lead | QH     | Hour   | Hour x QH |
|------|--------|--------|-----------|
| 12   | -0.4%  | -0.7%  | -0.1%     |
| 9    | +0.3%  | +1.8%  | +1.9%     |
| 6    | +0.7%  | +2.0%  | **+2.7%** |
| 3    | +1.0%  | +0.2%  | +2.2%     |
| 0    | +1.8%  | +0.4%  | +1.9%     |
| **Avg** | +0.7%  | +0.7%  | **+1.7%** |

## Recommendation
1. Train model WITHOUT `qh_position` feature
2. Learn Hour × QH interaction bias from calibration window (e.g., last 6 months)
3. Apply as post-hoc correction to predictions
4. Expected improvement: ~1.7-2.7% for mid-lead times

## Files
- `bias_calibration_v2.py` - Final experiment script
- `bias_calibration_v2_results.csv` - Results table
- `bias_calibration.png` - Visualization

## Why This Works
The QH position oscillation pattern (QH 1&3 over-predict, QH 2&4 under-predict) is a stable, systematic bias. By NOT including `qh_position` as a training feature, the model cannot learn this pattern directly. We then correct for it post-hoc using calibration data, which generalizes well to future predictions.
