# Signal Validation

Validation that the load surprise prediction signal works for trading.

## Key Finding

The signal remained profitable even after the pure arbitrage closed.

### Signal Performance (Dec 2025 - Jan 2026)

| Period | Always Sell | Signal-Based | Winner |
|--------|-------------|--------------|--------|
| Dec 2025 | +21,786 EUR/MWh | +1,582 EUR/MWh | Always Sell |
| Jan 2026 | +812 EUR/MWh | +3,102 EUR/MWh | **Signal** |

### Signal Accuracy

- **SELL signals** (pred < -100 MW): 80% profitable
- **BUY signals** (pred > +100 MW): 65% profitable

### Full Year 2025 Validation

The signal was tested over full 2025:
- Profitable in 10/12 months
- Retained 9.9% of total profit while reducing trades by 87%
- The signal worked, but was less efficient than "always sell" when arbitrage existed

## Files

| File | Description |
|------|-------------|
| `check_signal_jan2026.py` | Validates signal indicates direction in Jan 2026 |
| `check_signal_profit.py` | Compares signal vs always-sell profit |
| `analyze_jan2026_signal.py` | Deep analysis of Jan 2026 signal performance |
| `full_year_signal_test.py` | Full 2025 validation |
| `check_signal_transition.py` | Weekly transition analysis Dec-Jan |
| `13_signal_check_jan2026.png` | Direction prediction accuracy |
| `14_jan2026_signal_analysis.png` | Jan 2026 detailed analysis |
| `15_signal_validation_2025.png` | Full year validation |

## Conclusion

The signal became essential when the arbitrage closed. It identifies profitable trading opportunities by predicting load surplus/deficit conditions.
