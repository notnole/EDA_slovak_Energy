# Year Comparison Summary

## What This Analysis Contains

Comparison between 2024 (baseline) and 2025-2026 (current) to determine if historical data can be used for model training.

**Key Question**: Has the system behavior changed enough that 2024 data would hurt model performance?

---

## Key Findings

### 1. Aggregate Statistics Shifted

| Metric | 2024 | 2025-2026 | Change |
|--------|------|-----------|--------|
| Mean | +0.95 MWh | +2.24 MWh | +1.29 MWh |
| Std Dev | 12.50 MWh | 11.39 MWh | -1.11 MWh |
| Surplus % | 53.2% | 57.7% | +4.5% |

**Interpretation**: The grid is running with more generation margin in 2025-2026, and is slightly more stable.

### 2. Pattern Shapes Are Similar

The hourly and weekly patterns have the **same shape**, just shifted vertically:
- All hours show +0.3 to +2.9 MWh increase
- All days of week show +0.5 to +1.8 MWh increase
- The relative patterns (peak hours, weekend effect) are preserved

### 3. Microstructure Is Nearly Identical

This is the critical finding for ML:

| Metric | Correlation | Interpretation |
|--------|-------------|----------------|
| ACF | 0.998 | Identical memory structure |
| PACF | 0.998 | Identical direct correlations |
| Volatility ACF | 0.942 | Similar volatility clustering |
| Change dynamics | 0.995 | Identical period-to-period behavior |

**Rolling volatility**: 7.63 MWh in both periods (virtually identical)

**Period-to-period change std**: 8.59 vs 7.99 MWh (7% difference)

### 4. January Comparison (Same Month, Different Years)

| Year | January Mean | January Std |
|------|--------------|-------------|
| 2024 | +0.43 MWh | 8.35 MWh |
| 2025 | +3.64 MWh | 11.17 MWh |
| 2026 | -2.37 MWh | 13.63 MWh |

Note: January shows significant year-to-year variation, but this is normal for energy systems.

---

## Conclusion: Use All Data

**Recommendation: Include 2024 data in training**

### Why It's Safe

1. **Microstructure is identical** (r > 0.99 on ACF/PACF)
   - The time-series dynamics your model learns are the same
   - Lag features, rolling statistics will behave identically

2. **Your features are relative, not absolute**
   - Rolling mean captures local level → handles mean shift
   - Rolling std captures local volatility → identical across periods
   - HoD/DoW capture patterns → same shape in both periods

3. **More data helps edge cases**
   - 2024 has more extreme events (wider range)
   - Model needs exposure to rare scenarios

### Optional Safeguards

1. Add `year >= 2025` as a binary feature (probably unnecessary given microstructure similarity)
2. Use time-based CV: train on 2024 → validate on early 2025 → test on late 2025/2026
3. Monitor if `year` feature becomes important (would indicate regime difference)

---

## Implications for the Model

### Data Usage

- **Training set**: All of 2024 + 2025 (can use full history)
- **Validation**: Time-based split (e.g., last 2 months of training period)
- **Test**: Hold out recent data (e.g., 2026 January)

### Feature Engineering

The mean shift (+1.29 MWh) will be handled by:
- Rolling mean features (captures current local level)
- The model doesn't need to learn absolute values, just deviations from recent history

### No Special Handling Needed

- No need to demean or normalize by year
- No need for regime indicators
- No need to weight recent data more heavily

The dynamics are the same - only the operating point shifted.

---

## Files in This Folder

| File | Description |
|------|-------------|
| `01_distribution_comparison.png` | Histograms, boxplots, sign distribution |
| `02_daily_comparison.png` | Hourly patterns side-by-side |
| `03_weekly_comparison.png` | Day-of-week patterns side-by-side |
| `04_monthly_comparison.png` | Monthly patterns, January deep-dive |
| `05_time_series_comparison.png` | Full series, rolling means |
| `06_microstructure_comparison.png` | ACF, PACF, volatility clustering |
| `year_comparison_report.txt` | Detailed numerical comparison |
| `plot_descriptions.txt` | Text descriptions of each plot |
