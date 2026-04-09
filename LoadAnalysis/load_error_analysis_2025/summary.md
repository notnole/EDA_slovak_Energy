# Load Error Analysis - H2 2025

## Training Setup

| Component | Period | Samples |
|-----------|--------|---------|
| Stage 1 Training | 2024 (full year) | ~8,427 |
| Stage 2 Training | H1 2025 (Jan-Jun) | ~4,325 |
| Evaluation | H2 2025 (Jul-Dec) | ~4,140 |

**Key**: Stage 2 trained on out-of-fold (OOF) residuals - Stage 1 never saw 2025 data.

## Results Summary

| Horizon | Baseline MAE | Stage 1 MAE | Stage 2 MAE | Total Improv | S2 Gain | Direction Acc |
|---------|--------------|-------------|-------------|--------------|---------|---------------|
| **H+1** | 63.8 MW | 40.0 MW | **32.4 MW** | **+49.2%** | +19.0% | 83.5% |
| H+2 | 63.9 MW | 48.7 MW | 44.7 MW | +30.0% | +8.1% | 76.4% |
| H+3 | 64.0 MW | 54.0 MW | 52.2 MW | +18.3% | +3.3% | 71.3% |
| H+4 | 64.1 MW | 57.9 MW | 57.0 MW | +11.1% | +1.5% | 66.0% |
| H+5 | 64.2 MW | 60.2 MW | 60.2 MW | +6.3% | -0.1% | 63.0% |

## Key Findings

### H+1 Performance
- **49.2% improvement** over DAMAS baseline (32.4 vs 63.8 MW MAE)
- Stage 2 adds **+19% gain** over Stage 1 alone
- **83.5% direction accuracy** - model correctly predicts sign of error

### Performance Degradation with Horizon
- Expected decay from 49% (H+1) to 6% (H+5) improvement
- Stage 2 gain decreases from 19% to negligible at H+5
- Proper temporal decay confirms no data leakage

### Error Patterns (H+1)
- **Worst hours**: 23:00 (46 MW), 08:00 (43 MW), 19:00 (37 MW)
- **Bias**: -0.8 MW (slight under-prediction)
- **RMSE**: 42.3 MW

## Comparison with Jan 2026 Results

| Horizon | H2 2025 MAE | Jan 2026 MAE | Difference |
|---------|-------------|--------------|------------|
| H+1 | 32.4 MW | 34.0 MW | -1.6 MW (better) |
| H+2 | 44.7 MW | 46.1 MW | -1.4 MW (better) |
| H+3 | 52.2 MW | 53.4 MW | -1.2 MW (better) |
| H+4 | 57.0 MW | 58.3 MW | -1.3 MW (better) |
| H+5 | 60.2 MW | 61.7 MW | -1.5 MW (better) |

H2 2025 performance slightly better than Jan 2026, possibly due to:
- Larger test set (4,140 vs ~744 samples)
- No seasonal transition effects

## Artifacts

- `models/stage1_h{1-5}.joblib` - Stage 1 models
- `models/stage2_h{1-5}.joblib` - Stage 2 models
- `data/predictions_h{1-5}.parquet` - Test predictions
- `data/error_analysis.json` - Detailed analysis by hour/dow/month
- `plots/error_analysis_h{1-5}.png` - Visualization plots
