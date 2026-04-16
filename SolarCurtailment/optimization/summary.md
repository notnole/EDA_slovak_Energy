# Solar Curtailment Optimization: Two-Rule Grid Search

**Question**: What are the optimal thresholds for curtailing a 1MW solar panel that settles at the imbalance price, using (1) imbalance predictions and (2) DA market price as indicators?

## Setup

- **Plant**: 1 MW E-W solar, settling at imbalance price (not committed on DA)
- **Current rule**: Curtail when XGBoost imbalance prediction > 15 MWh
- **Proposed rules**:
  - Rule 1: Curtail if imbalance_pred > X (always)
  - Rule 2: Curtail if DA_price < A AND imbalance_pred > Y (Y < X)
  - Combined: curtail when either rule triggers

## Key Results

### Summer (Mar-Aug) - where curtailment matters

Using 2025 Mar-Aug data with simulated prediction noise (MAE = 9.0 MWh):

| Strategy | EUR/MWh | Curtailed | Uplift vs No Curtail |
|----------|---------|-----------|---------------------|
| No curtailment | 50.5 | 0% | -- |
| Current (pred > 15) | 58.1 | ~14% | +7.6 |
| **Best Rule 1 (pred > 26)** | **59.3** | **9.0%** | **+8.8** |
| **Best Combined (X=36, A=20, Y=8)** | **62.4** | **13.9%** | **+11.9** |
| Oracle (perfect foresight) | 67.1 | ~20% | +16.6 |

### Recommended Rules

```
Rule 1: Curtail when imbalance_prediction > 36 MWh
Rule 2: Curtail when DA_price < 20 EUR AND imbalance_prediction > 8 MWh
```

### Interpretation

- **Rule 1 alone** (pred > 26): Only curtails during extreme surplus. Safe but leaves money on the table. Already better than current pred > 15 because current threshold is too aggressive - it curtails periods that are actually profitable on average.
- **Adding Rule 2** (DA < 20 AND pred > 8): Adds +3.1 EUR/MWh over Rule 1 alone. When the DA price is very low (< 20 EUR), even moderate surplus predictions signal danger. This catches the moderate-surplus-but-still-negative-price events that Rule 1 misses.
- **Why current pred > 15 underperforms best Rule 1**: The current threshold is too low. It curtails many periods where imbalance price is still positive. The optimal Rule 1 threshold is higher (26-36 MWh) because the prediction has noise - you need a bigger signal to be confident.

### Winter (Oct-Mar) - curtailment has minimal value

With real predictions (Oct 2025 - Mar 2026):
- Baseline: 97.1 EUR/MWh, Best combined: 97.3 EUR/MWh
- Only 0.7% of periods worth curtailing
- **Recommendation: disable curtailment Nov-Feb**

## DA Price as Indicator

The heatmaps show clear value from the DA indicator:
- When DA < 20 EUR, the system is likely in surplus and negative imbalance prices are common
- The combined rule captures an additional ~5% of periods to curtail (beyond Rule 1) that have negative settlement prices
- The DA threshold is robust: anywhere from 15-30 EUR works well

## Data & Caveats

- Real predictions only cover Oct 2025 - Mar 2026 (winter/transition)
- Summer results use actual imbalance + simulated noise (MAE = 9.0 MWh from real prediction data)
- The XGBoost model may perform differently in summer (more solar events) - recalibrate with real summer data
- DA price is known day-ahead, so it's available at curtailment decision time

## Files

- `scripts/grid_search_v2.py` - Main optimization script
- `01_rule1_all_datasets.png` - Rule 1 sweep across all datasets
- `02_combined_heatmaps.png` - DA threshold vs imbalance threshold heatmaps
- `03_strategy_comparison.png` - Bar chart comparing all strategies
- `04_decision_space.png` - Scatter plot of decision space (pred vs DA, colored by settlement price)
- `data/summary_v2.csv` - Summary table
- `data/grid_summer_*.csv` - Full grid search results
