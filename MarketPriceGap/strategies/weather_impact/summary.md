# Weather Impact on DA-Imbalance Spread

## Cold Snap Analysis

- Cold snaps (daily mean <0C): 9 events found across 2 winters (2024/25 and 2025/26)
- Cold snap hours: mean spread -5.3 EUR/MWh, total P&L -6,392 EUR (vs +2.0 and +34,109 non-cold)
- Strategy loses in 5 of 8 cold snap periods
- Day-within-cold-snap pattern: days 1-2 mild losses, days 3-4 maximum pain (-36 to -50 EUR/MWh), days 6+ recovery

## Summer Heat Anomaly

- July/heat anomaly: solar duck curve drives midday spread to -43 EUR/MWh on hot days (>25C)
- Hot days (>25C mean): spread -20.7 EUR/MWh vs +2.4 normal days

## January 2026 Root Cause

- Cold snap -> load under-forecast (+17 MW bias) -> deficit (57% vs 45%) -> settlement price spikes
- Load error-imbalance correlation: r=-0.29 (Sep-Dec 2025) -> r=-0.62 (Jan 2026)

## Scripts

- `scripts/cold_snap_analysis.py` - cold snap identification and market impact
- `scripts/cold_snap_day_decay_analysis.py` - within-cold-snap day-by-day decay pattern
- `scripts/july_spread_analysis.py` - summer heat/solar duck curve analysis
- `deficit_root_cause.py`, `market_comparison.py` - Jan 2026 root cause investigation

## Charts

- `cold_snap_market_impact.png` - spread distribution during cold snaps
- `cold_snap_day_decay.png` - day-by-day P&L decay within cold snap events
- `july_spread_anomaly.png` - midday spread on hot vs normal days
- `deficit_root_cause_analysis.png` - load error to deficit to settlement cascade
- `da_vs_imbalance_2026.png` - DA vs imbalance price comparison 2026
- `da_vs_imbalance_sep25_vs_jan26.png` - regime shift Sep 2025 vs Jan 2026
