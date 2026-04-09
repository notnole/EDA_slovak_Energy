# Imbalance Nowcasting

## Overview

Predict 15-minute Slovak system imbalance (MWh) from real-time 3-minute SCADA regulation data. The production model (LightGBM V4) achieves 32% MAE improvement at Lead 12 (hardest) and 94% direction accuracy at Lead 0.

**Constraint**: Actual imbalance values are NOT available until next day. Model uses only real-time regulation, load, and time features.

## Final Model Performance (LightGBM V4)

| Lead Time | MAE (MWh) | Baseline MAE | Improvement | Direction Acc |
|-----------|-----------|--------------|-------------|---------------|
| 12 min | 4.50 | 6.64 | +32% | 78.9% |
| 9 min | 3.55 | 4.76 | +25% | 83.1% |
| 6 min | 2.73 | 3.35 | +18% | 86.7% |
| 3 min | 2.02 | 2.40 | +16% | 90.2% |
| 0 min | 1.30 | 2.03 | +36% | 94.3% |

## Sub-Analyses

### Data Exploration
| Folder | Description |
|--------|-------------|
| [label/](label/summary.md) | Imbalance distribution, seasonality, year comparison |
| [features/](features/summary.md) | SCADA feature analysis (correlation, lags, decomposition) |

### Models
| Folder | Description | Key Finding |
|--------|-------------|-------------|
| [models/baseline/](models/baseline/summary.md) | Deterministic baseline (-0.25 * mean(reg)) | Foundation for ML models |
| [models/lightgbm/](models/lightgbm/summary.md) | LightGBM v4 production model | 12-29% MAE reduction |
| [models/lightgbm/degradation_analysis/](models/lightgbm/degradation_analysis/summary.md) | Train on 2024, test on 2025 | Seasonal drift, not monotonic; retraining gives only +2.7% |
| [models/lightgbm/ar_correction/](models/lightgbm/ar_correction/summary.md) | Post-hoc bias correction | Hour x QH correction +1.7% avg improvement |
| models/ensemble/ | Ensemble experiments | Scripts + outputs available |

### Production Analysis
| Folder | Description |
|--------|-------------|
| analysis/production_features/ | Production type features (nuclear, import deviation) |
| [report/](report/00_executive_summary.md) | Full technical report (4 chapters) |

## Key Insights

1. **Proxy rolling mean** (avg of last 4 periods) is the #1 feature at Lead 12
2. **Baseline prediction** becomes dominant as lead time decreases
3. Simpler features beat complex ones -- V4 outperformed feature-heavy V5-V7
4. Retraining provides only +2.7% -- the irreducible error is ~4.5 MWh at Lead 12
5. Hour x QH bias correction adds +1.7% without retraining

### Solar Forecast Error (standalone)
| Folder | Description |
|--------|-------------|
| [analysis/solar_forecast_error/](analysis/solar_forecast_error/summary.md) | TSO solar forecast quality: MAE 10.9 MW, 88% of error is weather-driven |
