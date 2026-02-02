# Production Model Documentation

## Overview

This document describes the 5 production ML models for predicting electricity system imbalance (MWh) at different lead times before settlement end.

| Model | Lead Time | Prediction Time | Features |
|-------|-----------|-----------------|----------|
| `model_production_lt15.joblib` | 15 min | QH start | 4 |
| `model_production_lt12.joblib` | 12 min | QH start + 3 min | 5 |
| `model_production_lt9.joblib` | 9 min | QH start + 6 min | 5 |
| `model_production_lt6.joblib` | 6 min | QH start + 9 min | 3 |
| `model_production_lt3.joblib` | 3 min | QH start + 12 min | 3 |

---

## Model Specifications

### Model LT=15 (4 features)

**When to run:** At QH start (e.g., 10:00:00 for QH ending at 10:15:00)

**Required features:**
| Feature | Description | Calculation |
|---------|-------------|-------------|
| `roll_mean_5` | 15-min rolling mean | Mean of last 5 values (current + 4 lags) |
| `val_curr` | Current value | Latest available `live3minValue` (MW) |
| `roll_mean_10` | 30-min rolling mean | Mean of last 10 values |
| `roll_std_5` | 15-min rolling std | Std dev of last 5 values |

**Input array order:** `[roll_mean_5, val_curr, roll_mean_10, roll_std_5]`

---

### Model LT=12 (5 features)

**When to run:** At QH start + 3 min (e.g., 10:03:00 for QH ending at 10:15:00)

**Required features:**
| Feature | Description | Calculation |
|---------|-------------|-------------|
| `val_curr` | Current value | Latest available `live3minValue` (MW) |
| `roll_mean_5` | 15-min rolling mean | Mean of last 5 values |
| `lag_1` | Previous value | Value from 3 min ago |
| `roll_mean_10` | 30-min rolling mean | Mean of last 10 values |
| `roll_mean_20` | 60-min rolling mean | Mean of last 20 values |

**Input array order:** `[val_curr, roll_mean_5, lag_1, roll_mean_10, roll_mean_20]`

---

### Model LT=9 (5 features)

**When to run:** At QH start + 6 min (e.g., 10:06:00 for QH ending at 10:15:00)

**Required features:**
| Feature | Description | Calculation |
|---------|-------------|-------------|
| `val_curr` | Current value | Latest available `live3minValue` (MW) |
| `roll_mean_5` | 15-min rolling mean | Mean of last 5 values |
| `qh_cumsum` | QH cumulative sum | Sum of values in current QH so far |
| `lag_2` | Value 2 steps ago | Value from 6 min ago |
| `roll_mean_10` | 30-min rolling mean | Mean of last 10 values |

**Input array order:** `[val_curr, roll_mean_5, qh_cumsum, lag_2, roll_mean_10]`

---

### Model LT=6 (3 features)

**When to run:** At QH start + 9 min (e.g., 10:09:00 for QH ending at 10:15:00)

**Required features:**
| Feature | Description | Calculation |
|---------|-------------|-------------|
| `qh_cumsum` | QH cumulative sum | Sum of values in current QH so far |
| `val_curr` | Current value | Latest available `live3minValue` (MW) |
| `roll_mean_5` | 15-min rolling mean | Mean of last 5 values |

**Input array order:** `[qh_cumsum, val_curr, roll_mean_5]`

---

### Model LT=3 (3 features)

**When to run:** At QH start + 12 min (e.g., 10:12:00 for QH ending at 10:15:00)

**Required features:**
| Feature | Description | Calculation |
|---------|-------------|-------------|
| `qh_cumsum` | QH cumulative sum | Sum of values in current QH so far |
| `val_curr` | Current value | Latest available `live3minValue` (MW) |
| `roll_mean_5` | 15-min rolling mean | Mean of last 5 values |

**Input array order:** `[qh_cumsum, val_curr, roll_mean_5]`

---

## Feature Calculation Reference

### Data Grid
- Input signal: `live3minValue` (MW) on 3-minute intervals
- Grid alignment: `..., T-9, T-6, T-3, T (current)`

### Feature Formulas

```
val_curr    = value[T]
lag_1       = value[T-3min]
lag_2       = value[T-6min]

roll_mean_5  = mean(value[T], value[T-3], ..., value[T-12])     # 5 values, 15 min
roll_mean_10 = mean(value[T], value[T-3], ..., value[T-27])     # 10 values, 30 min
roll_mean_20 = mean(value[T], value[T-3], ..., value[T-57])     # 20 values, 60 min

roll_std_5   = std(value[T], value[T-3], ..., value[T-12])      # 5 values, 15 min

qh_cumsum    = sum of all values in current QH up to and including T
               (resets at each QH boundary: :00, :15, :30, :45)
```

### QH Cumsum Examples

For QH 10:00-10:15:
| Time | Values seen | qh_cumsum |
|------|-------------|-----------|
| 10:00 | v1 | v1 |
| 10:03 | v1, v2 | v1 + v2 |
| 10:06 | v1, v2, v3 | v1 + v2 + v3 |
| 10:09 | v1, v2, v3, v4 | v1 + v2 + v3 + v4 |
| 10:12 | v1, v2, v3, v4, v5 | v1 + v2 + v3 + v4 + v5 |

---

## Production Usage

### Loading Models

```python
import joblib
import numpy as np

# Load model and config
model_lt15 = joblib.load('models/model_production_lt15.joblib')
config = joblib.load('models/model_config.joblib')

# Check feature order
print(config['features'][15])  # ['roll_mean_5', 'val_curr', 'roll_mean_10', 'roll_std_5']
```

### Making Predictions

```python
def predict_lt15(val_curr, lag_1, lag_2, lag_3, lag_4,
                 lag_5, lag_6, lag_7, lag_8, lag_9):
    """
    Predict at LT=15 (at QH start).

    Args:
        val_curr: Current value (MW)
        lag_1..lag_9: Previous values going back 27 min

    Returns:
        Predicted systemImbalance (MWh)
    """
    values = [val_curr, lag_1, lag_2, lag_3, lag_4,
              lag_5, lag_6, lag_7, lag_8, lag_9]

    roll_mean_5 = np.mean(values[:5])
    roll_mean_10 = np.mean(values[:10])
    roll_std_5 = np.std(values[:5])

    features = np.array([[roll_mean_5, val_curr, roll_mean_10, roll_std_5]])
    return model_lt15.predict(features)[0]
```

```python
def predict_lt6(values_in_qh):
    """
    Predict at LT=6 (at QH start + 9 min).

    Args:
        values_in_qh: List of 4 values seen in QH so far [v1, v2, v3, v4]

    Returns:
        Predicted systemImbalance (MWh)
    """
    val_curr = values_in_qh[-1]
    qh_cumsum = sum(values_in_qh)

    # Need 5 values for roll_mean_5 (current + 4 from history)
    # Assume history is available
    roll_mean_5 = ...  # mean of last 5 values

    features = np.array([[qh_cumsum, val_curr, roll_mean_5]])
    return model_lt6.predict(features)[0]
```

---

## Timing Schedule

For a QH ending at `HH:MM:00`:

| Lead Time | Prediction Time | Model File |
|-----------|-----------------|------------|
| 15 min | HH:(MM-15):00 | model_production_lt15.joblib |
| 12 min | HH:(MM-12):00 | model_production_lt12.joblib |
| 9 min | HH:(MM-9):00 | model_production_lt9.joblib |
| 6 min | HH:(MM-6):00 | model_production_lt6.joblib |
| 3 min | HH:(MM-3):00 | model_production_lt3.joblib |

**Example for QH 10:15:00:**
| Lead Time | Prediction Time | Data Available Up To |
|-----------|-----------------|----------------------|
| 15 min | 10:00:00 | 10:00:00 |
| 12 min | 10:03:00 | 10:03:00 |
| 9 min | 10:06:00 | 10:06:00 |
| 6 min | 10:09:00 | 10:09:00 |
| 3 min | 10:12:00 | 10:12:00 |

---

## Output

- **Unit:** MWh (energy for 15-min settlement period)
- **Sign:** Positive = system long, Negative = system short
- **Range:** Typically -50 to +50 MWh (depends on market conditions)

---

## Model Performance

| Lead Time | Sign Accuracy | MAE (MWh) |
|-----------|---------------|-----------|
| 15 min | 78.3% | 4.27 |
| 12 min | 83.8% | 3.35 |
| 9 min | 87.4% | 2.53 |
| 6 min | 90.3% | 1.81 |
| 3 min | 93.5% | 1.13 |

**SignAccQH:** 89.3% (>=3/5 correct sign predictions per QH)
