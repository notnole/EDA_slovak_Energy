# DA vs IDM Price Collapse Analysis - January 2026

## Question

Can a BESS exploit DA-IDM price divergence? Specifically:
- Do extreme DA prices collapse on the IDM (buy cheap on IDM after selling high on DA)?
- Is there a systematic time-shift in peaks (buy DA at off-peak, sell IDM at shifted peak)?

## Finding: Mostly Inconclusive

The DA-IDM relationship in January 2026 shows no systematic, tradeable pattern. Both collapses and spikes occur, roughly cancelling out, and peak shifts are not predictable.

### IDM collapses (IDM << DA) - would favour BESS buy-on-IDM

- Jan 23: IDM drops 141 EUR below DA at morning peak (DA 350, IDM ~210)
- Jan 20: IDM -66 below DA at morning peak
- Jan 15: DA peaks at H8, IDM peak shifts +8h to H16 with lower amplitude

### IDM spikes (IDM >> DA) - goes against BESS strategy

- Jan 5: IDM explodes +194 above DA in the evening (IDM 377, DA 224)
- Jan 6: IDM peaks at H17 (374 EUR) while DA peaked at H9 (169). +8h shift, wrong direction.
- Jan 8: IDM +107 above DA at evening peak
- Jan 21: IDM +97 above DA

### Peak time-shifts

- 4 days with |shift| >= 4 hours (Jan 6, 15, 24: forward; Jan 2: backward)
- No consistent direction or magnitude
- DA peak hour is not a reliable predictor of IDM peak hour

## Why it's inconclusive

1. Collapses and spikes are roughly symmetric - some days IDM undershoots DA, other days it overshoots. No net bias large enough for a systematic strategy.
2. Peak shifts are inconsistent - sometimes forward, sometimes backward, mostly zero. Can't reliably predict which way the peak moves.
3. DA is a weak indicator of IDM divergence - the divergence appears driven by forecast errors discovered between DA and IDM gate closure, which are by definition unpredictable from DA data alone.
4. Small sample - January 2026 is one month. The patterns could be specific to this cold-weather period.

## What drives the divergence (when it happens)

The causal mechanism is well understood (see `temp_filter/cold_snap_causality.py`):
- Load forecast under-prediction during cold snaps -> traders discover the error -> IDM correction
- But the correction direction is inconsistent: sometimes IDM overshoots (panic buying), sometimes it undershoots (DA was already too high)

## Conclusion for BESS

The DA-IDM divergence is not a reliable additional signal for BESS scheduling beyond what the DA spread indicator already captures. The existing BESS simulation (192k EUR, 78% capture) uses DA-Imbalance spreads which already incorporate the settlement price outcome. Adding IDM timing arbitrage would not systematically improve results.

## Charts

- `jan2026_da_prices.png` - DA prices timeseries, January 2026
- `jan2026_da_vs_idm.png` - DA vs IDM VWAP overlay, full month
- `jan2026_extreme_da_idm.png` - 8 extreme DA days with IDM and seasonal baseline
- `jan2026_idm_anomalies.png` - 6 days with IDM collapses, spikes, and peak shifts
