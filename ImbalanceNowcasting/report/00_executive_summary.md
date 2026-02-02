# Imbalance Nowcasting Model - Executive Summary

## Project Objective

Develop a machine learning model to predict 15-minute system imbalance (MWh) using real-time 3-minute regulation data, enabling predictions at multiple lead times (12, 9, 6, 3, 0 minutes before settlement period ends).

## Key Constraint

**Actual imbalance values are NOT available until the next day.** The model can only use:
- Real-time regulation data (3-minute intervals)
- Real-time load data (3-minute intervals)
- Time features (hour, day of week)
- Proxy-based features derived from regulation

## Final Model: LightGBM V4

### Performance Summary

| Lead Time | V4 MAE | Baseline MAE | Improvement | Direction Accuracy |
|-----------|--------|--------------|-------------|-------------------|
| 12 min    | 4.50   | 6.64         | **+32%**    | 78.9%             |
| 9 min     | 3.55   | 4.76         | +25%        | 83.1%             |
| 6 min     | 2.73   | 3.35         | +18%        | 86.7%             |
| 3 min     | 2.02   | 2.40         | +16%        | 90.2%             |
| 0 min     | 1.30   | 2.03         | **+36%**    | 94.3%             |

### Key Achievements

1. **32% MAE improvement** at the hardest lead time (12 minutes)
2. **94.3% direction accuracy** at settlement end
3. **Robust across conditions**: Equal performance on weekdays/weekends, high/low load
4. **Real-time deployable**: Uses only data available before prediction time

## Methodology Overview

### Phase 1: Data Exploration
- Analyzed 72,375 imbalance observations (2024-2026)
- Confirmed no structural break between years (safe to use 2024 for training)
- Identified ~43% predictable variance from time patterns, ~57% requires ML

### Phase 2: Feature Engineering
- **Adopted**: Regulation, proxy lags, rolling statistics, time features, load deviation
- **Abandoned**: Production data (no correlation), actual imbalance lags (not available in real-time)
- Final model uses 35 features at lead 12, decreasing to 7 features at lead 0

### Phase 3: Model Development
- Tested 7 model versions with different feature sets
- V4 (real-time only features) outperformed versions using more complex features
- Simple, targeted features beat feature-heavy approaches

## Key Insights

1. **Proxy rolling mean** (average of last 4 periods' estimated imbalance) is the most important feature at lead 12
2. **Baseline prediction** becomes dominant as lead time decreases
3. **Low-importance features collectively matter**: Removing features with <1% individual importance degrades performance by 8-17%
4. **Error patterns**: Model performs best during midday (11-14h), worst during evening (21-23h)

## Recommendations

1. **Deploy V4 model** for production nowcasting
2. **Monitor hourly performance** - evening hours have higher uncertainty
3. **Re-train quarterly** to adapt to any system changes
4. **Consider ensemble** for critical applications (additional 2-3% improvement possible)

---

*Full technical details in subsequent sections.*
