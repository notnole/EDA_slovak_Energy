# Load Decomposition Analysis

## Overview
STL decomposition of Slovakia grid load to understand variance structure.

## Variance Decomposition (2024)

| Component | Variance Explained |
|-----------|-------------------|
| **Seasonal** | **57.0%** |
| **Trend** | **35.8%** |
| Residual | 7.0% |

## Key Findings

1. **Seasonality dominates** - 57% of load variance is from daily/weekly patterns
2. **Strong trend** - 36% from longer-term changes (seasonal, economic)
3. **Low residual** - Only 7% unexplained variance

## Weekday vs Weekend

| Metric | Weekday | Weekend | Difference |
|--------|---------|---------|------------|
| Mean load | 3,030 MW | 2,698 MW | **332 MW (12%)** |

Weekend loads are systematically lower with different daily profiles.

## Plots

- `01_stl_decomposition.png` - Full STL decomposition
- `01b_decomposition_weekday_weekend.png` - Weekday/weekend comparison

## Implications for Modeling

- Capturing seasonal patterns is **critical** (57% of variance)
- Weekday/weekend indicator is essential feature
- Residuals are small - good baseline achievable with seasonal model
