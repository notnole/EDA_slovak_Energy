# Price Decomposition

## Overview
STL decomposition of day-ahead electricity prices (2024 data).

## Variance Decomposition

| Component | Variance |
|-----------|----------|
| Trend | 23% |
| Seasonal | 44% |
| Residual | 33% |

## Key Finding
Prices are **much noisier** than load:
- Load residual: 7%
- Price residual: 33%

This makes price prediction significantly harder.

## Plots
- `01_price_decomposition.png` - STL decomposition with 168h (weekly) period
