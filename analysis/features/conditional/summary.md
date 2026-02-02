# Conditional Analysis Summary

## Purpose
Analyze how the regulation-imbalance correlation changes under different conditions to understand when our model will perform better or worse.

## Baseline
- Overall regulation-imbalance correlation: **r = -0.667**
- Total samples: 71,198

## Results

### Correlation Stability by Condition

| Condition | Min r | Max r | Range | Stable? |
|-----------|-------|-------|-------|---------|
| Weekend | -0.667 | -0.666 | 0.001 | YES |
| Load Level | -0.690 | -0.654 | 0.036 | YES |
| Regulation Sign | -0.581 | -0.545 | 0.036 | YES |
| Load Ramp | -0.683 | -0.636 | 0.047 | YES |
| Month | -0.742 | -0.574 | 0.168 | NO |
| Hour | -0.785 | -0.459 | 0.326 | NO |
| **Imbalance Magnitude** | **-0.776** | **-0.133** | **0.644** | **NO** |

### Key Findings

#### 1. Weekday/Weekend: STABLE ✓
No significant difference. Model will perform equally on weekdays and weekends.

#### 2. Hour of Day: VARIABLE
| Period | Hours | Avg Correlation |
|--------|-------|-----------------|
| Best | 11-14 | -0.77 |
| Good | 2-4, 9-10 | -0.70 to -0.78 |
| Worst | 21-23 | -0.46 to -0.53 |

Model performs best during late morning/early afternoon, worst during late evening.

#### 3. Month: VARIABLE
| Month | Correlation |
|-------|-------------|
| Best: April | -0.742 |
| Worst: February | -0.574 |

Spring months (Apr-May) show strongest correlations.

#### 4. Load Level: STABLE ✓
No significant difference across load quartiles. Model works equally well during high and low demand periods.

#### 5. Imbalance Magnitude: HIGHLY VARIABLE ⚠️
| Imbalance Size | Avg |Imb| (MWh) | Correlation |
|----------------|---------------------|-------------|
| Small | 1.0 | -0.133 |
| Medium-Small | 3.4 | -0.306 |
| Medium-Large | 7.4 | -0.532 |
| Large | 19.9 | -0.776 |

**Critical finding**: Regulation is a much weaker predictor for small imbalances.
- Large imbalances (|imb| > 10 MWh): Well predicted (r = -0.78)
- Small imbalances (|imb| < 2 MWh): Poorly predicted (r = -0.13)

#### 6. Load Ramp: STABLE ✓
No significant difference between rising and falling load periods.

#### 7. Regulation Sign: STABLE ✓
Similar correlation for positive (oversupply) and negative (undersupply) regulation.

## Model Implications

### Performance Expectations
1. **Best performance**: Large imbalances during hours 11-14
2. **Worst performance**: Small imbalances during hours 21-23
3. **Equal performance**: Weekday/weekend, high/low load

### Evaluation Strategy
- Evaluate model separately for:
  - Large vs small imbalances
  - Peak vs off-peak hours
- Report MAE separately for different imbalance magnitude bins

### Feature Engineering
- Consider adding `hour_of_day` as feature (correlation varies significantly)
- `is_weekend` less important (correlation stable)
- Consider interaction: `regulation × hour` may capture time-varying relationship

## Files Generated
- `data/01_weekend_analysis.csv`
- `data/02_hour_analysis.csv`
- `data/03_month_analysis.csv`
- `data/04_load_level_analysis.csv`
- `data/05_imbalance_magnitude_analysis.csv`
- `data/06_load_ramp_analysis.csv`
- `data/07_regulation_sign_analysis.csv`
- `data/conditional_summary.csv`
- `data/conditional_report.txt`
