# Plant Operations EDA - Bardejov CHP

## Data

- **Source**: `data/Bardejov/plant_timeseries.csv` (extracted from DHS sheets in Kalkulacie Excel files)
- Three columns: heat_load_kW (heat to network), cooling_kW (heat dissipated), electricity_kW (CHP generation)
- 152,943 records, 2020-01 to 2025-12 (hourly Jan-Feb, 15-min Mar-Dec)
- 97.2% exact match with existing heat_load_timeseries.csv

## Data extraction notes

The Excel files have two formats:
- **2020-2024 (and 2025 Jan-Apr)**: Instantaneous kW values, columns correctly labeled
- **2025 May-Dec**: Cumulative MWh counters with **swapped column labels** (REGELKUEHLKREIS = heat to network, LEISTUNG-NETZ = total thermal). Required differencing, column swap, and start-of-interval timestamp alignment.

## Key Findings

### CHP operation mode changes

The plant has gone through significant operational changes:

| Period | Mode | Electricity | Cooling | Notes |
|--------|------|-------------|---------|-------|
| 2020-2021 | Full CHP | ~8 MW constant | 5-20 MW (inverse of heat demand) | Turbine at full capacity |
| 2022 | Reduced CHP | Variable, declining | Reduced | Transition period |
| 2023 | Partial CHP | ~4-6 MW when running | Variable | Intermittent operation |
| 2024 Jan-Apr | Heat-only | 0 MW | ~0 MW | Turbine offline |
| 2024 May-Dec | Full CHP | ~6-8 MW | 10-20 MW | Turbine restarted |
| 2025 | Variable CHP | ~3-6 MW | Variable | Ongoing |

### Energy balance

When CHP is running, total thermal output is roughly constant (~20-24 MW):
- **Heat demand high** (winter peak): most goes to network, cooling near zero
- **Heat demand low** (summer/night): excess dumped through cooling circuit

This means: `total_thermal = heat_to_network + cooling ≈ constant` when CHP is on.

### Annual energy (approximate, depends on data coverage)

| Year | Heat (GWh) | Cooling (GWh) | Electricity (GWh) | Days |
|------|-----------|--------------|-------------------|------|
| 2021 | 78.1 | 49.2 | 33.0 | 365 |
| 2022 | 60.3 | 14.0 | 18.5 | 334 |
| 2023 | 67.6 | 24.5 | 18.2 | 365 |
| 2024 | 66.0 | 92.0 | 43.1 | 365 |
| 2025 | 70.3 | 18.4 | 10.4 | 357 |

### Implications for heat load forecasting

1. **Heat demand is independent of CHP mode**: The heat_to_network column captures actual district heating demand regardless of whether the turbine is running
2. **Cooling is the balancing variable**: When CHP runs, the plant produces constant thermal output and dumps excess to the cooling circuit
3. **Electricity generation constrains flexibility**: When the turbine runs, the plant cannot reduce thermal output below ~20 MW even if heat demand is only 2 MW (summer)

## Plots

1. `01_daily_energy_breakdown.png` - Stacked daily energy: heat, cooling, electricity + utilization ratio
2. `02_monthly_energy_by_component.png` - Monthly energy by component, colored by year
3. `03_electricity_vs_heat.png` - Scatter: electricity and cooling vs heat demand, by year
4. `04_winter_week_all_components.png` - Hourly detail: Jan 2021 winter week with all three components
5. `05_chp_operation_timeline.png` - Monthly average electricity generation over time

## Scripts

- `scripts/plot_plant_operations.py` - All 5 plots + summary statistics
