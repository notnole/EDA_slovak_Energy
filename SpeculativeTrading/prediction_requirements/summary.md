# Prediction Requirement Analysis: Decision Matrix

## Goal: EUR 200/day from speculative electricity trading

## Key Question: What accuracy do we need for each prediction target?

## Decision Matrix

| Target | Breakeven | EUR/MWh @60% | EUR/MWh @70% | MWh @70% | Sharpe @70% |
|--------|-----------|-------------|-------------|----------|------------|
| T1: DA->IDM | 50% | +8.2 | +9.3 | 1 | 11.4 |
| T2: IDM 6h->1h | 95% | -2.7 | -1.7 | >999 | -3.1 |
| T3: IDM dir 3h->1h | 95% | -5.9 | -5.2 | >999 | -11.0 |
| T4: IDM round-trip | 64% | -12.6 | +1.5 | 5 | 2.8 |
| T5: Imb price->IDM | 50% | +30.8 | +35.6 | 0 | 21.6 |
| T6: Imb direction | 50% | +35.7 | +39.4 | 0 | 24.8 |
| T7: DA-IDM spread dir | 50% | +8.8 | +9.8 | 1 | 12.1 |

## Interpretation

- **Breakeven accuracy**: Minimum prediction accuracy needed to be profitable after bid-ask costs
- **EUR/MWh @X%**: Average profit per MWh traded at X% accuracy
- **MWh @70%**: Position size (MWh per hour) needed for EUR 200/day at 70% accuracy
- **Sharpe @70%**: Annualized Sharpe ratio at that position size

## Recommendations

Targets are ranked by feasibility (low breakeven + low position size + high Sharpe):

1. **T6: Imb direction** -- Breakeven: 50%, FEASIBLE
2. **T5: Imb price->IDM** -- Breakeven: 50%, FEASIBLE
3. **T7: DA-IDM spread dir** -- Breakeven: 50%, FEASIBLE
4. **T1: DA->IDM** -- Breakeven: 50%, FEASIBLE
5. **T4: IDM round-trip** -- Breakeven: 64%, FEASIBLE
6. **T2: IDM 6h->1h** -- Breakeven: 95%, UNPROFITABLE
7. **T3: IDM dir 3h->1h** -- Breakeven: 95%, UNPROFITABLE