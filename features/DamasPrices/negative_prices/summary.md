# Negative Prices Analysis

## Overview
Analysis of negative electricity prices in the Slovak day-ahead market.

## Statistics

| Metric | Value |
|--------|-------|
| Frequency | 2.48% of hours |
| Mean negative | ~-30 EUR/MWh |
| Min (most negative) | -202.70 EUR/MWh |

## Timing Patterns

| Pattern | Finding |
|---------|---------|
| **By hour** | Most common at midday (11-15) |
| **By day** | 78% on weekends |
| **By month** | More frequent in spring/summer |

## Cause
Negative prices indicate **oversupply**, typically:
- High renewable generation (solar at midday)
- Low weekend demand
- Grid export constraints

## Net Import During Negative Prices
Higher net imports during negative prices = Slovakia receiving cheap/excess power from neighbors.

## Plots
- `05_negative_prices.png` - Hourly/monthly distribution, net import relationship
