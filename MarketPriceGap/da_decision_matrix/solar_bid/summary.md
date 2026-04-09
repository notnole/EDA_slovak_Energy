# Solar Farm DA Bid Optimization

**Question**: What minimum DA bid price maximizes revenue for a 1MW E-W solar farm in Slovakia?

## Setup

- **Plant**: 1 MW solar, half east / half west facing, 35 deg tilt, Bratislava (48.15N)
- **Production**: PVGIS-calibrated (v5.3), 684 MWh spring-summer (943 MWh/year)
- **Data**: 2025 full year + 2026 Jan-Mar
- **Volume predictor**: Predicts system imbalance in MWh. Positive = surplus (too much generation, prices crash). Does NOT predict price directly.

## Key Result

| Metric | Value |
|--------|-------|
| **Recommended strategy** | **Hybrid with volume-predictor curtailment** |
| **Optimal DA bid (summer)** | **58 EUR/MWh** |
| Revenue per MWh (optimal) | 62.2 EUR/MWh |
| vs Pure DA (bid=0) | +11.4 EUR/MWh (+22.5%) |
| vs DA-or-nothing | +9.1 EUR/MWh (+17.1%) |
| Season total (685 MWh) | 42,581 EUR |

## How It Works

Each hour during solar production:

1. **DA price >= bid threshold** -> SELL on DA (committed delivery)
2. **DA price < threshold** -> check the volume predictor:
   - **Large surplus predicted** -> CURTAIL (avoid negative prices)
   - **Deficit or small surplus** -> PRODUCE at imbalance price

### What "curtail top 50% surplus volume" means

The volume predictor tells us the system imbalance in MWh. When the DA market rejects our bid (price too low), we look at the predictor:

- Rank all rejected hours by predicted surplus (largest `imbalance_mwh` first)
- Budget curtailing up to **50% of the volume** that would go to imbalance
- Allocate this budget to the hours with the **biggest predicted surplus** first
- Only curtail during actual surplus hours (imbalance_mwh > 0)

The 50% refers to MWh of production, not number of hours. We sacrifice half the imbalance-routed volume to avoid the worst surplus hours.

## Why 58 EUR/MWh?

Two forces set the threshold:

1. **Below 58 EUR, imbalance + curtailment wins.** The volume predictor avoids enough negative-price hours to make imbalance settlement profitable. Curtailed hours average -42 EUR (avoided losses). This only works when NOT committed on DA.

2. **Above 58 EUR, DA certainty wins.** DA prices above 58 EUR average 112+ EUR/MWh, systematically beating imbalance. The commitment is worth the lost curtailment optionality.

### Why 50% curtailment budget?

| Curtailment level | Meaning | Opt Bid | Revenue/MWh |
|-------------------|---------|---------|-------------|
| 0% | No curtailment, all imbalance | 84 EUR | 51.3 |
| 10% | Curtail top 10% surplus vol | 84 EUR | 60.9 |
| 25% | Curtail top 25% surplus vol | 64 EUR | 61.5 |
| **50%** | **Curtail top 50% surplus vol** | **58 EUR** | **62.2** |
| 75% | Curtail top 75% surplus vol | 48 EUR | 61.5 |
| 100% | DA or nothing (curtail 100%) | 0 EUR | 53.1 |

50% is the sweet spot: enough to dodge most negative-price hours, but not so much that we waste good imbalance hours. The first 10% captures most of the value; 50% squeezes out the rest.

### Volume predictor vs price oracle

The predictor uses volume (imbalance_mwh) as a proxy for price:
- **Oracle** (perfect price foresight): curtails hours averaging **-119 EUR**
- **Volume predictor**: curtails hours averaging **-42 EUR** at 50% budget
- Prediction gap: ~3,000 EUR/MW/season left on the table
- A better predictor (wind/solar forecasts, cross-border flows) could close this gap

## Seasonal Rules

| Season | Months | Optimal Bid | Rationale |
|--------|--------|-------------|-----------|
| **Winter** | Nov-Feb | 0 EUR (pure DA) | DA consistently beats imbalance, no negative price risk |
| **Transition** | Mar-Apr | 40-55 EUR | Few surplus events, hybrid adds 4-6 EUR/MWh |
| **Peak solar** | May-Jun | 45-58 EUR | Surplus events sharply increase, hybrid adds 10-15 EUR/MWh |
| **Late summer** | Jul-Aug | 40-55 EUR | Shorter days, less surplus, hybrid adds 8-12 EUR/MWh |

2026 Feb-Mar confirms the winter rule: DA dominates at 96.3 EUR/MWh.

## Solar Profile

PVGIS-calibrated 1MW E-W (0.5MW East + 0.5MW West, 35 deg tilt):
- Annual: 943 MWh, Spring-summer: 684 MWh
- Double-peak pattern: ~420 kW at 9-10h and 14-15h, dip to ~385 kW at noon
- E-W advantage: more production in morning/evening when prices are higher and surplus is lower
- Validated against PVGIS v5.3 monthly totals (match within 1-2% for May-Jul)

## Files

- `solar_bid_recommendation.py` - Analysis script (all figures + recommendation)
- `01_revenue_composition.png` - **Where energy goes and money comes from** vs DA bid threshold (stacked areas)
- `02_strategy_comparison.png` - Revenue curves for all curtailment levels
- `03_seasonal_strategy.png` - Monthly optimal bids + monthly revenue comparison
- `04_current_season.png` - 2026 vs 2025 Feb-Mar price scatter and hourly profiles
- `05_volume_predictor.png` - Curtailment value (oracle vs volume predictor) + produce/curtail scatter
- `06_production_economics.png` - E-W solar profile by month + monthly revenue & production
- `recommendation_summary.csv` - Key numbers
