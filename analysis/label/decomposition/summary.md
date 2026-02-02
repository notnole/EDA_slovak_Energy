# Decomposition Analysis Summary

## What This Analysis Contains

Time series decomposition to understand what portion of the signal is predictable from simple patterns vs what requires ML.

**Methods used**:
- STL (Seasonal-Trend decomposition using LOESS) - daily seasonality
- MSTL (Multiple STL) - daily + weekly seasonality
- Spectral/Fourier analysis - frequency domain

---

## Key Findings

### 1. Variance Decomposition (MSTL)

| Component | Variance Explained | What It Represents |
|-----------|-------------------|---------------------|
| Trend | 6.2% | Slow drift in operating point |
| Daily Seasonal | 19.6% | 24-hour repeating pattern |
| Weekly Seasonal | 8.3% | 7-day repeating pattern |
| **Residual** | **56.8%** | Unpredictable component |

**~43% is "easy"** - captured by simple time features
**~57% is "hard"** - requires model to learn from other signals

### 2. Std Deviation Reduction

| Stage | Std Dev | Reduction |
|-------|---------|-----------|
| Original | 11.96 MWh | - |
| After removing daily seasonal | 9.50 MWh | 20.6% |
| After removing daily + weekly | 9.02 MWh | 24.6% |

Even perfect seasonal features leave ~9 MWh of unexplained variation.

### 3. Dominant Frequencies (Spectral Analysis)

| Period | Meaning | Power Rank |
|--------|---------|------------|
| ~670 samples | ~1 week | 1st |
| 4 samples | 1 hour | 2nd |
| 96 samples | 24 hours | 5th |
| 2 samples | 30 min | 6th |

Strong sub-hourly patterns exist - the 3-min input data can capture useful high-frequency dynamics.

### 4. Extracted Seasonal Pattern

**Daily pattern**:
- Peak: 15:00 (+4.8 MWh)
- Trough: 20:00 (-5.4 MWh)
- Amplitude: ~10 MWh swing

This pattern is consistent across the dataset.

---

## Critical Insight: Decomposition ≠ Prediction

### What Decomposition Does NOT Tell Us

1. **Cannot extrapolate trend** - Trend is fitted with hindsight using future data
2. **Cannot decompose future points** - We'd need to know the value first
3. **Residual is not a training target** - Can't compute residual without knowing original

### What Decomposition DOES Tell Us

1. **Validates feature choices**:
   - Daily seasonal (20%) → HoD feature will help
   - Weekly seasonal (8%) → DoW feature will help
   - Trend (6%) → Rolling mean will approximate

2. **Quantifies difficulty**:
   - Best possible RMSE from seasonality alone: ~9 MWh
   - Model must beat this to add value

3. **Shows residual has structure**:
   - Residual ACF shows autocorrelation
   - Lag features and rolling stats will help capture this

---

## Implications for the Model

### Feature Engineering Priority

| Priority | Feature Type | Captures | Expected Contribution |
|----------|--------------|----------|----------------------|
| **High** | `hour_of_day` | Daily seasonal | ~20% variance |
| **High** | Rolling mean (recent 3-min) | Local trend | ~6% + edge info |
| **High** | Lag features | Residual autocorrelation | Part of 57% |
| **Medium** | `day_of_week` | Weekly seasonal | ~8% variance |
| **Medium** | Rolling std | Local volatility | Uncertainty estimate |
| **Low** | `month` | Annual cycle | Small, may overfit |

### What the Model Actually Learns

The model does NOT learn "trend + seasonal + residual" separately.

It learns: **"Given these features, predict the target"**

The decomposition just proves these features contain the right information:
- HoD encodes daily pattern implicitly
- Rolling mean encodes local level implicitly
- The model finds the mapping automatically

### Expected Performance Bounds

| Baseline | RMSE |
|----------|------|
| Naive (use last value) | ~8.3 MWh |
| Perfect seasonal model | ~9.0 MWh |
| Good ML model (target) | <8.0 MWh |

Note: Naive baseline is surprisingly good because consecutive values are correlated. The challenge is beating this with features.

### Nowcasting-Specific Insight

For nowcasting (predicting 15-min total from partial 3-min data):

- **Cumulative sum within QH** is your strongest feature
- At minute 12 (4th observation), you've seen 80% of the quarter-hour
- Early predictions (minute 3-6) rely more on seasonal patterns
- Late predictions (minute 12) rely more on extrapolating current trajectory

---

## Files in This Folder

| File | Description |
|------|-------------|
| `01_stl_decomposition.png` | Trend + daily seasonal + residual |
| `02_mstl_decomposition.png` | Trend + daily + weekly + residual |
| `03_spectral_analysis.png` | Frequency domain, dominant periods |
| `04_variance_decomposition.png` | Variance explained by each component |
| `05_seasonal_patterns.png` | Extracted daily and weekly patterns |
| `decomposition_report.txt` | Detailed numerical results |
