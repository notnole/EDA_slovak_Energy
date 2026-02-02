# Bias Features Test

## Purpose
Test if adding DAMAS systematic bias features improves predictions.

## Features Tested
- `prev_day_mean_error` - Previous day's average error
- `daily_roll_3d`, `daily_roll_7d` - Rolling daily error averages
- `same_day_running_mean` - Cumulative same-day bias
- `early_day_bias` - Morning hours (0-6) average error
- `sign_match_24h` - Error sign persistence

## Results
- **Gain: -6.9%** (actually hurt performance!)
- Reason: Collinear with existing features (error lags already capture this)

## Conclusion
Bias features don't help because:
1. Error lags already capture recent bias patterns
2. Bias patterns are not stable across years (2024 train vs 2025 test)

## Files
- `bias_features_test.py` - Bias feature implementation and testing
