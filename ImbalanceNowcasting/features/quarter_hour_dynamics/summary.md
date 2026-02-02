# Within Quarter-Hour Dynamics Summary

## Purpose
Analyze how features evolve during the 5 observations (minutes 0, 3, 6, 9, 12) within each 15-minute settlement period, and how early observations predict the final imbalance.

## Key Question
**Can we predict the 15-min imbalance before the period ends?**

## Results

### 1. Predictive Power by Minute

| Minute | Single Obs r | Cumulative Mean r | R² |
|--------|-------------|-------------------|-----|
| 0 | -0.685 | -0.685 | 47% |
| 3 | -0.682 | -0.708 | 50% |
| 6 | -0.822 | -0.782 | 61% |
| 9 | -0.873 | -0.849 | 72% |
| 12 | -0.895 | -0.889 | 79% |

**Key finding**: Even at minute 0, we can predict 47% of variance. By minute 12, we reach 80%.

### 2. Nowcasting Improvement
- **First observation (min 0)**: r = -0.685
- **Full period (min 12)**: r = -0.895
- **Improvement**: +30% correlation gain

### 3. Within-QH Variability Features

| Feature | Correlation | Use? |
|---------|-------------|------|
| QH Mean | -0.889 | YES - best single feature |
| QH Trend (last-first) | -0.260 | Maybe - captures direction |
| QH Std | -0.015 | NO - not predictive |
| QH Range | -0.014 | NO - not predictive |

### 4. Early Warning Capability
Regulation clearly separates for extreme vs normal imbalances even at minute 0:
- **Extreme negative imbalance**: Regulation ~+50 to +75 MW (positive = oversupply)
- **Extreme positive imbalance**: Regulation ~-65 to -90 MW (negative = undersupply)
- **Normal**: Regulation ~-5 to -10 MW

## Model Implications

### For Nowcasting (5 predictions per label):
1. **Prediction 1 (at min 3)**: Use reg_min0, expect R² ~ 47%
2. **Prediction 2 (at min 6)**: Use mean(reg_min0, reg_min3), expect R² ~ 50%
3. **Prediction 3 (at min 9)**: Use cumulative mean, expect R² ~ 61%
4. **Prediction 4 (at min 12)**: Use cumulative mean, expect R² ~ 72%
5. **Prediction 5 (at min 15)**: Use full QH mean, expect R² ~ 79%

### Recommended Features
1. **Cumulative mean of regulation** - best overall feature
2. **Latest regulation observation** - captures most recent state
3. **Minute indicator** - model should know how much data is available
4. **QH trend** (optional) - small additional signal

### What NOT to use
- QH std / range - no predictive power
- Individual minute features beyond cumulative mean

## Conversion: MW to MWh

Note: Regulation is in MW, imbalance is in MWh.
- Regulation (MW) × (15 min / 60) = Energy contribution (MWh)
- So: reg_MW × 0.25 ≈ imbalance_MWh (approximately)
- The strong negative correlation confirms: regulation ≈ -imbalance
