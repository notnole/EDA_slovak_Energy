# Price-Load Correlation

## Overview
Analysis of relationship between day-ahead prices and grid load.

## Overall Correlation

| Relationship | Correlation |
|--------------|-------------|
| Price vs Actual Load | 0.36 |
| Price vs Forecast Load | 0.36 |
| Price vs Load Error | ~0.05 |

## Correlation by Hour

| Hour | Correlation |
|------|-------------|
| Hour 14 (peak) | 0.74 (highest) |
| Night hours | ~0.10 (lowest) |

## Key Finding
- Price-load correlation strongest during peak demand hours
- Weaker correlation at night when base load dominates
- Load forecast error has minimal correlation with price

## Implication for Modeling
DAMAS likely already incorporates price information. Adding price features to load prediction provides limited value.

## Plots
- `04_price_load_correlation.png` - Scatter plots, hourly correlation, heatmap
