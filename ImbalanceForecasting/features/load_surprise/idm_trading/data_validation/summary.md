# Data Validation

Scripts to validate data correctness and trading logic.

## Validation Results

All data and calculations were verified to be correct:

1. **Columns**: Correct price columns used
   - IDM: "Weighted average price of all trades (EUR/MWh)"
   - Imbalance: "Imbalance Settlement Price (EUR/MWh)"

2. **Datetime alignment**: Verified correct
   - IDM Period 1 = 00:00-00:15 = Imbalance datetime 00:00:00

3. **Trading logic**: Verified correct
   - Spread = IDM - Imbalance
   - Positive spread = profit when selling on IDM

4. **Extreme values**: Real market events
   - IDM: -427 to +724 EUR/MWh
   - Imbalance: -8,009 to +5,167 EUR/MWh

## Files

| File | Description |
|------|-------------|
| `validate_data.py` | Comprehensive data validation |
| `check_alignment.py` | Datetime alignment verification |
| `check_extreme_prices.py` | Extreme price analysis |
| `check_trading_logic.py` | Trading logic verification |
| `check_price_levels.py` | Price level analysis by period |

## Conclusion

The data and calculations are correct. The market convergence observed in Jan 2026 is real, not a data artifact.
