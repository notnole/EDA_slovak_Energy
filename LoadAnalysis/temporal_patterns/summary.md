# Temporal Patterns Analysis

## Overview
Analysis of hourly, daily, weekly, and monthly load patterns.

## Hourly Profile

| Day Type | Peak Hour | Peak Load | Off-Peak Hour | Off-Peak Load |
|----------|-----------|-----------|---------------|---------------|
| Weekday | Hour 10 | 3,333 MW | Hour 4 | 2,462 MW |
| Weekend | Hour 12 | 2,995 MW | Hour 4 | 2,341 MW |

Peak-to-trough ratio: ~35% daily variation.

## Weekly Pattern

| Day | Mean Load (MW) |
|-----|---------------|
| Monday | 3,050 |
| Tuesday | 3,080 |
| Wednesday | 3,065 |
| Thursday | 3,050 |
| Friday | 2,980 |
| Saturday | 2,720 |
| Sunday | 2,675 |

Clear weekday/weekend distinction.

## Monthly Pattern

| Season | Months | Avg Load |
|--------|--------|----------|
| Winter | Dec-Feb | ~3,200 MW |
| Spring | Mar-May | ~2,900 MW |
| Summer | Jun-Aug | ~2,700 MW |
| Autumn | Sep-Nov | ~3,000 MW |

Seasonal amplitude: ~500 MW

## Basic Statistics

| Metric | Value |
|--------|-------|
| Mean | 2,936 MW |
| Std | 455 MW |
| Min | 1,919 MW |
| Max | 4,170 MW |
| Load factor | 70.4% |

## Plots

- `02_temporal_patterns.png` - Hourly, daily, weekly patterns
- `02b_load_heatmap.png` - Hour x Day heatmap
- `03_distributions.png` - Load distribution histograms
- `07_load_duration_curve.png` - Load duration curve
