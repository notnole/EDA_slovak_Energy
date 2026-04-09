"""EDA of Bardejov CHP plant operations: electricity generation and heat dissipation."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "Bardejov"
PLANT_FILE = DATA_DIR / "plant_timeseries.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1]

# Load data
print("[*] Loading plant timeseries...")
df = pd.read_csv(PLANT_FILE, parse_dates=["datetime"])
df = df.set_index("datetime").sort_index()

# Resample to hourly for cleaner plots
hourly = df.resample("h").mean()
hourly = hourly.dropna(subset=["heat_load_kW"])

# Convert to MW
for col in ["heat_load_kW", "cooling_kW", "electricity_kW"]:
    hourly[col + "_MW"] = hourly[col] / 1000

print(f"[+] {len(hourly):,} hourly records, {hourly.index.min()} to {hourly.index.max()}")

# Daily aggregation
daily = hourly.resample("D").agg(
    heat_MWh=("heat_load_kW", lambda x: x.mean() * 24 / 1000),
    cool_MWh=("cooling_kW", lambda x: x.mean() * 24 / 1000 if x.notna().any() else np.nan),
    elec_MWh=("electricity_kW", lambda x: x.mean() * 24 / 1000 if x.notna().any() else np.nan),
)
daily = daily.dropna(subset=["heat_MWh"])
daily["total_thermal_MWh"] = daily["heat_MWh"] + daily["cool_MWh"].fillna(0)

# Monthly aggregation
monthly = daily.resample("MS").agg(
    heat_GWh=("heat_MWh", lambda x: x.sum() / 1000),
    cool_GWh=("cool_MWh", lambda x: x.sum() / 1000),
    elec_GWh=("elec_MWh", lambda x: x.sum() / 1000),
    total_thermal_GWh=("total_thermal_MWh", lambda x: x.sum() / 1000),
)

# Print summary stats
print("\n--- Summary by year ---")
yearly = daily.groupby(daily.index.year).agg(
    heat_GWh=("heat_MWh", lambda x: x.sum() / 1000),
    cool_GWh=("cool_MWh", lambda x: x.sum() / 1000),
    elec_GWh=("elec_MWh", lambda x: x.sum() / 1000),
    days=("heat_MWh", "count"),
)
yearly["total_thermal_GWh"] = yearly["heat_GWh"] + yearly["cool_GWh"]
yearly["elec_efficiency_%"] = yearly["elec_GWh"] / (yearly["total_thermal_GWh"] + yearly["elec_GWh"]) * 100
for y, row in yearly.iterrows():
    print(f"  {y}: heat={row['heat_GWh']:.1f} GWh, cool={row['cool_GWh']:.1f} GWh, "
          f"elec={row['elec_GWh']:.1f} GWh, days={row['days']:.0f}")

# ============================================================
# Plot 1: Time series - all three components stacked
# ============================================================
print("\n[*] Plot 1: Daily energy time series...")
fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

ax = axes[0]
ax.fill_between(daily.index, daily["heat_MWh"], alpha=0.7, color="firebrick", label="Heat to Network")
ax.fill_between(daily.index, daily["heat_MWh"],
                daily["heat_MWh"] + daily["cool_MWh"].fillna(0),
                alpha=0.5, color="steelblue", label="Heat Dissipated (Cooling)")
ax.set_ylabel("Daily Energy (MWh)")
ax.set_title("Thermal Output Breakdown")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_ylim(0, None)

ax = axes[1]
ax.fill_between(daily.index, daily["elec_MWh"].fillna(0), alpha=0.7, color="goldenrod")
ax.set_ylabel("Daily Electricity (MWh)")
ax.set_title("Electricity Generation (CHP Turbine)")
ax.grid(True, alpha=0.3)
ax.set_ylim(0, None)

ax = axes[2]
# Ratio: useful heat / total thermal
ratio = daily["heat_MWh"] / daily["total_thermal_MWh"].clip(lower=1)
ax.scatter(daily.index, ratio * 100, s=2, alpha=0.4, color="darkgreen")
ax.axhline(100, color="black", linewidth=0.5, linestyle="--")
ax.set_ylabel("Heat Utilization (%)")
ax.set_title("Heat to Network / Total Thermal Output")
ax.set_ylim(0, 110)
ax.grid(True, alpha=0.3)

for a in axes:
    a.xaxis.set_major_locator(mdates.YearLocator())
    a.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

fig.suptitle("Bardejov CHP Plant Operations - Daily Energy", fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "01_daily_energy_breakdown.png", dpi=150)
print("[+] Saved 01_daily_energy_breakdown.png")

# ============================================================
# Plot 2: Monthly energy bars by year
# ============================================================
print("[*] Plot 2: Monthly energy by component...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, (col, title, color) in zip(axes, [
    ("heat_GWh", "Heat to Network", "firebrick"),
    ("cool_GWh", "Heat Dissipated", "steelblue"),
    ("elec_GWh", "Electricity Generated", "goldenrod"),
]):
    for year in sorted(monthly.index.year.unique()):
        ym = monthly[monthly.index.year == year]
        ax.plot(ym.index.month, ym[col], "o-", markersize=4, label=str(year), alpha=0.7)
    ax.set_xlabel("Month")
    ax.set_ylabel("Energy (GWh)")
    ax.set_title(title)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)

fig.suptitle("Bardejov CHP - Monthly Energy by Component", fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "02_monthly_energy_by_component.png", dpi=150)
print("[+] Saved 02_monthly_energy_by_component.png")

# ============================================================
# Plot 3: Electricity vs Heat scatter by year
# ============================================================
print("[*] Plot 3: Electricity vs heat load scatter...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Daily scatter colored by year
for year in sorted(daily.index.year.unique()):
    yd = daily[daily.index.year == year]
    ax1.scatter(yd["heat_MWh"], yd["elec_MWh"], s=10, alpha=0.5, label=str(year))
ax1.set_xlabel("Heat to Network (MWh/day)")
ax1.set_ylabel("Electricity Generated (MWh/day)")
ax1.set_title("Daily: Electricity vs Heat Demand")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Daily scatter: cooling vs heat
for year in sorted(daily.index.year.unique()):
    yd = daily[daily.index.year == year]
    ax2.scatter(yd["heat_MWh"], yd["cool_MWh"], s=10, alpha=0.5, label=str(year))
ax2.set_xlabel("Heat to Network (MWh/day)")
ax2.set_ylabel("Heat Dissipated (MWh/day)")
ax2.set_title("Daily: Cooling vs Heat Demand")
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.suptitle("Bardejov CHP - Operational Relationships", fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "03_electricity_vs_heat.png", dpi=150)
print("[+] Saved 03_electricity_vs_heat.png")

# ============================================================
# Plot 4: Hourly profiles - winter week with all components
# ============================================================
print("[*] Plot 4: Winter week with all components...")
fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

# Pick a winter week in 2021 when CHP was running at full capacity
week = hourly["2021-01-11":"2021-01-17"]

ax = axes[0]
ax.plot(week.index, week["heat_load_kW_MW"], color="firebrick", linewidth=1, label="Heat to Network")
ax.plot(week.index, week["cooling_kW_MW"], color="steelblue", linewidth=1, label="Cooling (Dissipated)")
total = week["heat_load_kW_MW"] + week["cooling_kW_MW"].fillna(0)
ax.plot(week.index, total, color="gray", linewidth=1, linestyle="--", alpha=0.7, label="Total Thermal")
ax.set_ylabel("Power (MW)")
ax.set_title("Thermal Output - Winter Week (Jan 2021, CHP Running)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, None)

ax = axes[1]
ax.plot(week.index, week["electricity_kW_MW"], color="goldenrod", linewidth=1.5)
ax.set_ylabel("Electricity (MW)")
ax.set_title("Electricity Generation")
ax.grid(True, alpha=0.3)
ax.set_ylim(0, None)

ax = axes[2]
util = week["heat_load_kW_MW"] / total.clip(lower=0.1) * 100
ax.plot(week.index, util, color="darkgreen", linewidth=1)
ax.axhline(100, color="black", linewidth=0.5, linestyle="--")
ax.set_ylabel("Heat Utilization (%)")
ax.set_title("% of Thermal Output Used for Heating")
ax.set_ylim(0, 110)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b"))

fig.suptitle("Bardejov CHP - Winter Week Detail (Jan 11-17, 2021)", fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "04_winter_week_all_components.png", dpi=150)
print("[+] Saved 04_winter_week_all_components.png")

# ============================================================
# Plot 5: CHP operation mode changes over time
# ============================================================
print("[*] Plot 5: CHP operation mode timeline...")
fig, ax = plt.subplots(figsize=(16, 5))

# Monthly average electricity
monthly_elec = hourly["electricity_kW_MW"].resample("MS").mean()
monthly_elec = monthly_elec.dropna()

ax.bar(monthly_elec.index, monthly_elec.values, width=25, color="goldenrod", alpha=0.7)
ax.set_ylabel("Avg Electricity Generation (MW)")
ax.set_title("CHP Turbine Operation Over Time")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(0, None)

# Annotate key transitions
ax.axvline(pd.Timestamp("2022-01-01"), color="red", linestyle="--", alpha=0.5)
ax.text(pd.Timestamp("2022-02-01"), ax.get_ylim()[1] * 0.9, "CHP reduced",
        fontsize=9, color="red", alpha=0.7)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "05_chp_operation_timeline.png", dpi=150)
print("[+] Saved 05_chp_operation_timeline.png")

plt.close("all")
print("[+] All done.")
