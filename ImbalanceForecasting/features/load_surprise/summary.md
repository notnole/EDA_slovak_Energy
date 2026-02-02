# Load Surprise Effect on Imbalance

## Folder Structure

```
load_surprise/
├── basic_analysis/       # Load surprise vs imbalance correlation
├── direction_confidence/ # Direction prediction thresholds
├── qh_position/          # Quarter-hour position effects
├── predicted_surprise/   # H+1/2/3 prediction analysis
├── price_impact/         # Price impact of predictions
├── idm_trading/          # IDM arbitrage strategy
└── summary.md            # This documentation

Each subfolder contains:
  - Python script(s)
  - PNG visualizations
  - data/ subfolder with CSV outputs
```

## Definition

**Load Surprise = Actual Hourly Load - DAMAS Day-Ahead Forecast**

- Positive surprise: Demand higher than predicted (under-forecast)
- Negative surprise: Demand lower than predicted (over-forecast)

## Key Findings

### Day vs Night Comparison

| Period | Correlation (r) | R-squared | N samples |
|--------|-----------------|-----------|-----------|
| **Day (5-22)** | **-0.3138** | **0.0985** | 49,619 |
| Night (22-5) | -0.2513 | 0.0632 | 20,456 |
| Overall | -0.3045 | 0.0927 | 70,075 |

**Key Insight**: Day hours show 1.2x stronger correlation than night
(-0.314 vs -0.251).

### Hourly Analysis

| Metric | Hour | Value |
|--------|------|-------|
| Strongest correlation | 11:00 | r = -0.390 |
| Weakest correlation | 19:00 | r = -0.188 |
| Highest variability | 12:00 | std = 109 MW |

### By Load Surprise Magnitude (Day Hours)

| Magnitude | N Samples | Correlation | Imbalance Std |
|-----------|-----------|-------------|---------------|
| <50 MW | 20,707 | -0.085 | 12.0 MWh |
| 50-100 MW | 15,484 | -0.249 | 12.6 MWh |
| 100-200 MW | 11,600 | -0.440 | 14.0 MWh |
| 200-500 MW | 1,704 | -0.583 | 18.2 MWh |

### Direction Effect (Day Hours)

| Direction | N Samples | Mean Imbalance |
|-----------|-----------|----------------|
| Under-forecast (+) | 22,596 | -1.39 MWh |
| Over-forecast (-) | 27,023 | +4.88 MWh |

## Physical Interpretation

1. **Negative correlation** (r = -0.314):
   - Higher-than-expected load (positive surprise) --> More negative imbalance
   - When demand exceeds forecast, system tends to be SHORT (negative imbalance)

2. **Causality chain**:
   - DAMAS forecast sets day-ahead schedules
   - If actual load > forecast, generation is insufficient
   - TSO activates upward regulation --> negative imbalance

3. **Day vs Night**:
   - Day: More load variability, harder to forecast
   - Night: Stable baseload, smaller forecast errors

## Model Implications

1. **Load surprise explains ~9.8% of imbalance variance** (day hours)
   - Compare to regulation which explains ~45%
   - Useful as secondary feature, not primary predictor

2. **Strongest effect at large surprises**:
   - |surprise| > 200 MW: r = -0.582756012039182
   - Consider threshold-based feature

3. **Hour interaction**:
   - Peak hours (11:00) show strongest effect
   - Evening (19:00) shows weakest

## Files Generated

See individual subfolders for scripts, visualizations, and data files.

## Direction Confidence Analysis

### Key Thresholds (Day Hours 5-22)

| Confidence | Predict POSITIVE Imb | Predict NEGATIVE Imb |
|------------|---------------------|---------------------|
| 60% | < -62 MW | > +88 MW |
| 65% | < -62 MW | > +138 MW |
| 70% | < -112 MW | > +162 MW |
| 75% | < -138 MW | > +212 MW |
| 80% | < -238 MW | > +238 MW |

### Interpretation

- **Load Surprise < -125 MW**: 70%+ chance of POSITIVE imbalance (system long)
- **Load Surprise > +150 MW**: 70%+ chance of NEGATIVE imbalance (system short)
- **Load Surprise between -75 and +75 MW**: Low confidence (<65%), direction uncertain

### Physical Meaning

1. **Negative load surprise** (actual < forecast):
   - Less demand than scheduled
   - Generation exceeds consumption
   - System tends to be LONG (positive imbalance)

2. **Positive load surprise** (actual > forecast):
   - More demand than scheduled
   - Consumption exceeds generation
   - System tends to be SHORT (negative imbalance)

### Trading Implications

For imbalance direction betting:
- Wait for |load surprise| > 125-150 MW for 70%+ confidence
- Smaller surprises have too much noise for reliable direction prediction


## Quarter-Hour Position Effect

The hourly DAMAS forecast creates different effects across the 4 QHs within each hour.

### Correlation by QH Position

| QH | Time | Correlation | Interpretation |
|----|------|-------------|----------------|
| QH1 | :00-:15 | -0.329 | Strong - surprise just starting |
| QH2 | :15-:30 | -0.349 | Strongest - full surprise effect |
| QH3 | :30-:45 | -0.315 | Weakening - system reacting |
| QH4 | :45-:60 | -0.265 | Weakest - reserves activated |

### Direction Prediction Accuracy (at 70% thresholds)

| QH | Predict POSITIVE | Predict NEGATIVE |
|----|------------------|------------------|
| QH1 | 79.5% | 75.9% |
| QH2 | 81.7% | 78.0% |
| QH3 | 72.6% | 78.2% |
| QH4 | 72.4% | 70.8% |

### Key Insight

**First half (QH1-2) is more predictable than second half (QH3-4)**

- Correlation: -0.34 vs -0.29
- Accuracy: ~80% vs ~72% for positive prediction
- The TSO activates balancing reserves during the hour, reducing the load surprise effect

### Trading Implication

If using load surprise for direction prediction:
- **Prefer QH1 and QH2** settlements - higher confidence
- **Be cautious with QH3-4** - effect is diluted


## Predicted Load Surprise vs Future Imbalance

Using the 5-hour nowcasting model predictions to forecast imbalance direction.

### Correlation: Predicted Load Surprise vs Future Imbalance

| Horizon | Correlation | R-squared | N samples |
|---------|-------------|-----------|-----------|
| H+1 | -0.3013 | 0.0908 | 11,949 |
| H+2 | -0.2680 | 0.0718 | 11,949 |
| H+3 | -0.2354 | 0.0554 | 11,948 |
| H+1 (actual error) | -0.3420 | 0.1170 | 17,518 |
| H+2 (actual error) | -0.3420 | 0.1170 | 17,517 |
| H+3 (actual error) | -0.3421 | 0.1170 | 17,516 |

### Direction Prediction Accuracy

**Predict NEGATIVE imbalance when predicted load surprise > threshold:**

| Horizon | > +50 MW | > +100 MW | > +150 MW | > +200 MW |
|---------|----------|-----------|-----------|-----------|
| H+1 | 59.0% | 68.3% | 77.9% | 89.0% |
| H+2 | 58.4% | 67.7% | 77.7% | 82.2% |
| H+3 | 56.9% | 68.8% | 78.8% | 90.6% |

**Predict POSITIVE imbalance when predicted load surprise < threshold:**

| Horizon | < -50 MW | < -100 MW | < -150 MW | < -200 MW |
|---------|----------|-----------|-----------|-----------|
| H+1 | 72.9% | 79.0% | 80.0% | 85.1% |
| H+2 | 71.9% | 77.0% | 78.1% | 81.2% |
| H+3 | 70.9% | 77.2% | 77.5% | N/A |

### Key Findings

1. **Predicted load surprise correlates with future imbalance** at all horizons
2. **Correlation decreases with horizon** (as expected - prediction error increases)
3. **Direction prediction is possible** but accuracy decreases with horizon
4. **Actual (realized) load surprise has stronger correlation** than predicted


## Price Impact Analysis (H+2 Predictions)

Using the 2-hour ahead load surprise predictions to analyze price impact.

### Average Imbalance Price by Prediction Direction

| Threshold | Deficit Price | Surplus Price | Price Spread | Deficit Acc | Surplus Acc |
|-----------|---------------|---------------|--------------|-------------|-------------|
| +/-50 MW | 115.1 EUR | 62.0 EUR | +53.0 EUR | 58.4% | 71.9% |
| +/-100 MW | 130.6 EUR | 46.2 EUR | +84.4 EUR | 67.7% | 77.0% |
| +/-150 MW | 156.8 EUR | 25.0 EUR | +131.8 EUR | 77.7% | 78.1% |
| +/-200 MW | 162.1 EUR | -72.7 EUR | +234.8 EUR | 82.2% | 81.2% |

**Baseline average price**: 90.0 EUR/MWh

### Key Findings

1. **Significant price asymmetry**: When H+2 prediction indicates deficit (surprise > +100 MW),
   average price is higher than baseline. Surplus predictions have lower prices.

2. **Price spread increases with threshold**: Higher thresholds give larger price differences.

3. **Trading implication**: If confident about deficit (system short), prices tend to be higher.
   If surplus (system long), prices tend to be lower than average.

4. **Accuracy vs sample trade-off**: Higher thresholds give better accuracy but fewer signals.

### Profit Potential

At the +/-100 MW threshold:
- Deficit signals: Higher prices (system short, needs upward regulation)
- Surplus signals: Lower prices (system long, needs downward regulation)

This suggests value in timing trades based on the H+2 load surprise prediction.


## IDM Arbitrage Analysis (Sep-Dec 2025)

### Key Finding

**There is a structural arbitrage between IDM and Imbalance prices.**
The prediction correctly indicates imbalance direction, but it doesn't matter -
you profit by always selling on IDM regardless of prediction.

### Market Statistics

| Metric | Value |
|--------|-------|
| Avg IDM Price | 114.8 EUR/MWh |
| Avg Imbalance Price | 94.9 EUR/MWh |
| **Avg Spread** | **20.0 EUR/MWh** |
| Positive Spread Rate | 68% |

### Strategy Comparison

| Strategy | N Trades | Total EUR/MWh | Avg EUR/MWh | Win Rate |
|----------|----------|---------------|-------------|----------|
| Always Sell | 4,129 | 82,374 | 20.0 | 68% |
| Prediction < -50 MW | 844 | 11,199 | 13.3 | 63% |
| Prediction < -100 MW | 256 | 4,147 | 16.2 | 61% |

### Interpretation

1. **Prediction works for direction**: Negative predicted surprise correctly indicates
   positive imbalance (system long), and vice versa.

2. **But spread is positive everywhere**: IDM prices are systematically higher than
   imbalance prices across ALL prediction bins.

3. **Prediction filters OUT good trades**: By only trading when surprise < -50 MW,
   we miss profitable trades in other conditions.

4. **Simple strategy wins**: "Always sell on IDM, settle at imbalance" beats any
   prediction-based filtering.

### Why Does This Arbitrage Exist?

Possible explanations:
- IDM participants are risk-averse and willing to pay a premium for certainty
- Imbalance settlement has penalties/risks not reflected in the price
- Market participants overestimate imbalance price volatility
- Liquidity differences between IDM and imbalance settlement

### Trading Implication

For QH1-2 during day hours (5:00-22:00):
- **Sell on IDM, let it settle for imbalance**
- Expected profit: ~20 EUR/MWh (Sep-Dec 2025)
- Win rate: ~68%
- No prediction needed
