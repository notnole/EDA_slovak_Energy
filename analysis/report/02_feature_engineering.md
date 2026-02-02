# Feature Engineering Decisions

## Overview

This document explains which features were adopted for the final model, which were abandoned, and the rationale behind each decision.

---

## 1. Features ADOPTED

### 1.1 Core Regulation Features

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `reg_cumulative_mean` | Mean of available regulation observations | Primary predictor (r = -0.67). Cumulative mean outperforms individual observations. |
| `baseline_pred` | -0.25 × weighted regulation | Encapsulates the deterministic relationship. Becomes dominant at shorter lead times. |

**Evidence**: Correlation analysis showed regulation explains 45% of variance alone.

### 1.2 Proxy-Based Lag Features

Since actual imbalance is not available until the next day, we use a **proxy**:
```
proxy = -0.25 × mean(regulation)
```

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `proxy_lag1` | Previous period's proxy | Captures temporal autocorrelation (imbalance at lag 1 has r = 0.76) |
| `proxy_lag2`, `proxy_lag3`, `proxy_lag4` | Older lags | Diminishing returns but collectively useful |
| `proxy_rolling_mean4` | Rolling mean of last 4 proxies | **Most important feature at lead 12** (25.9% importance) |
| `proxy_rolling_std4` | Rolling std of last 4 proxies | Captures recent volatility regime |
| `proxy_rolling_mean10`, `proxy_rolling_std10` | Longer rolling windows | Extended context for longer leads |

**Evidence**: Lag correlation analysis showed adding lag1 improves R² by +2.4%.

![Lag Correlation Decay](../features/lag_correlation/data/regulation_lag_correlation.csv)

### 1.3 Historical Regulation Features

Statistics computed from the last 10-20 three-minute observations BEFORE the current settlement period.

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `reg_hist_mean_10` | Mean of last 10 observations (30 min) | Captures recent regulation level |
| `reg_hist_std_10` | Std of last 10 observations | Captures recent volatility |
| `reg_hist_trend_10` | Last obs minus 10 obs ago | Direction of regulation drift |
| `reg_hist_mean_20` | Mean of last 20 observations (1 hour) | Extended historical context |
| `reg_momentum` | Change from 2 obs ago to 1 obs ago | Recent acceleration |

**Evidence**: These features capture information not in the current period's observations, especially useful at lead 12 where only 1 observation is available.

### 1.4 Time Features

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `hour_sin`, `hour_cos` | Cyclical hour encoding | Captures daily pattern (19.6% of variance). Cyclical encoding avoids discontinuity at midnight. |
| `is_weekend` | Binary weekend flag | Captures +1.46 MWh weekend effect |
| `dow_sin`, `dow_cos` | Cyclical day-of-week | Captures weekly pattern (8.3% of variance) |

**Evidence**: Decomposition showed daily + weekly patterns explain ~28% of variance.

**Note**: Hour features used cyclical (sin/cos) encoding rather than categorical to:
1. Reduce feature count
2. Capture smooth daily pattern
3. Allow gradient-based learning

### 1.5 Load Features

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `load_deviation` | Current load minus expected load at same time-of-day | Small independent signal (r = -0.12), adds +0.3% R² |

**Evidence**: Load deviation slightly better than raw load. Deviation removes predictable daily pattern, leaving "surprise" component.

### 1.6 Sign-Based Features

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `proxy_last_sign` | Sign of previous proxy (+1/0/-1) | 78% persistence probability - sign tends to continue |
| `proxy_consecutive_same_sign` | Count of consecutive same-sign periods | Captures momentum |
| `proxy_prop_positive_4` | Proportion positive in last 4 periods | Recent regime indicator |

**Evidence**: Advanced analysis showed strong sign persistence (78% continuation probability).

### 1.7 Volatility Features

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `proxy_volatility_ratio` | Short-term std / Long-term std | Detects volatility regime shifts |
| `proxy_high_volatility` | Binary flag when ratio > 1.5 | Simple regime indicator |

**Evidence**: ARCH effects analysis showed strong volatility clustering.

---

## 2. Features ABANDONED

### 2.1 Production Data

| Feature | Why Abandoned |
|---------|---------------|
| `production_mw` | Near-zero correlation with imbalance (r = -0.05) |
| `production_deviation` | Same issue (r = -0.04) |

**Evidence**: Correlation analysis showed R² = 0.25% - essentially no predictive power.

**Additional reason**: Limited data coverage (only from Oct 2025), would require imputation for earlier periods.

### 2.2 Export/Import Data

| Feature | Why Abandoned |
|---------|---------------|
| `export_import_mw` | Near-zero correlation (r = +0.02) |
| `export_import_deviation` | Same issue (r = +0.006) |

**Evidence**: R² = 0.06% - no predictive power.

### 2.3 Actual Imbalance Lags

| Feature | Why Abandoned |
|---------|---------------|
| `imbalance_lag1` | **NOT AVAILABLE IN REAL-TIME** |
| `imbalance_lag2`, etc. | Actual imbalance only published next day |

**Critical Constraint**: We initially experimented with actual imbalance lags (V3 model) but this was INVALID for production use. The proxy-based features were developed as the real-time alternative.

### 2.4 Raw Load (vs Deviation)

| Feature | Why Abandoned |
|---------|---------------|
| `load_mw` | Deviation is better (r = -0.116 vs -0.101) |

**Rationale**: Load deviation removes the predictable daily pattern, leaving the "surprise" component which is what matters for imbalance.

### 2.5 Month Feature

| Feature | Why Abandoned |
|---------|---------------|
| `month` | Risk of overfitting to seasonal artifacts |

**Rationale**: Monthly variation is modest (r varies by ~0.17 across months). Including month as a categorical feature risks overfitting to training year patterns that may not generalize.

### 2.6 Within-Period Std/Range (at Lead 12)

| Feature | Why Abandoned |
|---------|---------------|
| `reg_std` | Zero importance at lead 12 (only 1 observation) |
| `reg_range` | Same issue |

**Rationale**: These features require 2+ observations within the period. At lead 12, only minute 0 is available, so std = range = 0.

### 2.7 Regulation Deviation (ToD-adjusted)

| Feature | Why Abandoned |
|---------|---------------|
| `regulation_deviation` | Trend too noisy to extrapolate (44% residual) |

**Rationale**: Unlike load (which is 96% predictable), regulation is 56% unpredictable. The "deviation from expected" adds noise rather than signal.

---

## 3. Feature Selection by Lead Time

The final model uses different feature sets for different lead times, following the principle: **use more features when less current-period information is available**.

| Lead Time | Features | Rationale |
|-----------|----------|-----------|
| **12 min** | 35 features | Only 1 observation available - need historical context |
| **9 min** | 28 features | 2 observations - can use within-period stats |
| **6 min** | 15 features | 3 observations - less history needed |
| **3 min** | 10 features | 4 observations - current period dominates |
| **0 min** | 7 features | Full period - minimal features needed |

### Lead 12 Feature List (35 features)

**Core**: `baseline_pred`, `reg_cumulative_mean`

**Historical Regulation**: `reg_hist_mean_10`, `reg_hist_std_10`, `reg_hist_trend_10`, `reg_hist_min_10`, `reg_hist_max_10`, `reg_hist_range_10`, `reg_hist_mean_20`, `reg_hist_std_20`, `reg_momentum`, `reg_acceleration`

**Proxy Lags**: `proxy_lag1`, `proxy_lag2`, `proxy_lag3`, `proxy_rolling_mean4`, `proxy_rolling_std4`, `proxy_rolling_mean10`, `proxy_rolling_std10`

**Sign Features**: `proxy_last_sign`, `proxy_last_positive`, `proxy_consecutive_same_sign`, `proxy_prop_positive_4`, `proxy_prop_positive_10`

**Proxy Momentum**: `proxy_momentum`, `proxy_acceleration`, `proxy_deviation_from_mean`

**Volatility**: `proxy_volatility_ratio`, `proxy_high_volatility`

**Time**: `hour_sin`, `hour_cos`, `is_weekend`, `dow_sin`, `dow_cos`

**Other**: `load_deviation`

### Lead 0 Feature List (7 features)

**Core**: `baseline_pred`, `reg_cumulative_mean`

**Within-period**: `reg_std`, `reg_range`, `reg_trend`

**Lag**: `proxy_lag1`

**Sign**: `proxy_last_sign`

---

## 4. Feature Importance Results

### Lead 12 Minutes (Top 10)

| Rank | Feature | Importance % |
|------|---------|--------------|
| 1 | `proxy_rolling_mean4` | 25.9% |
| 2 | `reg_cumulative_mean` | 15.6% |
| 3 | `baseline_pred` | 15.1% |
| 4 | `proxy_lag1` | 14.8% |
| 5 | `reg_hist_mean_20` | 7.8% |
| 6 | `reg_hist_mean_10` | 2.6% |
| 7 | `load_deviation` | 1.8% |
| 8 | `hour_cos` | 1.7% |
| 9 | `hour_sin` | 1.4% |
| 10 | `reg_hist_max_10` | 1.3% |

**Key Insight**: The top 4 features account for ~71% of importance. However, removing the remaining features (each <1%) degrades performance by 13%, demonstrating the "long tail" effect.

### Lead 0 Minutes (All 7)

| Rank | Feature | Importance % |
|------|---------|--------------|
| 1 | `baseline_pred` | 54.9% |
| 2 | `reg_cumulative_mean` | 39.6% |
| 3 | `reg_trend` | 3.7% |
| 4 | `proxy_lag1` | 0.6% |
| 5 | `reg_std` | 0.6% |
| 6 | `reg_range` | 0.6% |
| 7 | `proxy_last_sign` | 0.0% |

**Key Insight**: At lead 0, the baseline prediction and cumulative mean dominate (94.5% combined) - historical features become less relevant when full current-period data is available.

---

## 5. Lessons Learned

### What Worked
1. **Proxy features**: Successfully replaced unavailable actual imbalance lags
2. **Rolling statistics**: Captured temporal patterns and volatility regimes
3. **Lead-time-specific feature sets**: Optimized for available information
4. **Cyclical time encoding**: Better than categorical for gradient boosting

### What Didn't Work
1. **Production/Export data**: No correlation, limited coverage
2. **Complex error-correction features (V6, V7)**: Added noise, not signal
3. **Too many features**: V6 with 74 features performed worse than V4 with 35

### Key Principle
**Targeted simplicity beats complexity**. The best-performing model (V4) used carefully selected features based on domain knowledge, not maximum feature count.

---

*See Section 3 for model development and iteration history.*
