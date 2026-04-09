"""Exploratory data analysis of Bardejov heat load time series."""

import calendar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from pathlib import Path

# Paths
DATA_FILE = Path(__file__).resolve().parents[4] / "data" / "Bardejov" / "heat_load_timeseries.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1]

# Load and resample to hourly MW
print("[*] Loading heat load data...")
df = pd.read_csv(DATA_FILE, parse_dates=["datetime"])
df = df.set_index("datetime").sort_index()
hourly_MW = df["heat_load_kW"].resample("h").mean() / 1000

# Exclude the 2025 Mar-Apr shutdown (zeros distort stats)
shutdown_mask = (hourly_MW.index >= "2025-03-13") & (hourly_MW.index <= "2025-04-30")
hourly_clean = hourly_MW[~shutdown_mask]
print(f"[+] {len(hourly_clean):,} hourly readings (excl. 2025 shutdown)")

# ============================================================
# 1. Monthly boxplots
# ============================================================
print("[*] Plot 1: Monthly boxplots...")
fig, ax = plt.subplots(figsize=(14, 6))
month_data = [hourly_clean[hourly_clean.index.month == m].dropna().values for m in range(1, 13)]
bp = ax.boxplot(month_data, patch_artist=True, whis=(5, 95),
                medianprops=dict(color="black", linewidth=1.5))

# Color by season
season_colors = {
    "winter": "#c0392b",    # red
    "shoulder": "#e67e22",  # orange
    "summer": "#2980b9",    # blue
}
month_seasons = ["winter", "winter", "shoulder", "shoulder", "summer", "summer",
                 "summer", "summer", "shoulder", "shoulder", "winter", "winter"]
for patch, season in zip(bp["boxes"], month_seasons):
    patch.set_facecolor(season_colors[season])
    patch.set_alpha(0.6)

ax.set_xticklabels([calendar.month_abbr[m] for m in range(1, 13)])
ax.set_ylabel("Heat Load (MW)")
ax.set_title("Bardejov Heat Load - Monthly Distribution (2021-2025)")
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "01_monthly_boxplots.png", dpi=150)
print("[+] Saved 01_monthly_boxplots.png")

# ============================================================
# 2. Load duration curve
# ============================================================
print("[*] Plot 2: Load duration curve...")
sorted_MW = np.sort(hourly_clean.dropna().values)[::-1]
pct = np.linspace(0, 100, len(sorted_MW))

fig, ax = plt.subplots(figsize=(12, 6))
ax.fill_between(pct, sorted_MW, alpha=0.3, color="firebrick")
ax.plot(pct, sorted_MW, linewidth=1, color="firebrick")

# Annotate key levels
for level_pct in [10, 25, 50, 75, 90]:
    idx = int(level_pct / 100 * len(sorted_MW))
    val = sorted_MW[min(idx, len(sorted_MW) - 1)]
    ax.annotate(f"{val:.1f} MW", xy=(level_pct, val),
                xytext=(level_pct + 3, val + 1.5),
                fontsize=8, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))
    ax.axhline(val, color="gray", linestyle=":", alpha=0.3)

ax.set_xlabel("% of Hours Exceeded")
ax.set_ylabel("Heat Load (MW)")
ax.set_title("Bardejov Heat Load - Load Duration Curve")
ax.set_xlim(0, 100)
ax.set_xticks(range(0, 101, 5))
ax.set_ylim(0, None)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "02_load_duration_curve.png", dpi=150)
print("[+] Saved 02_load_duration_curve.png")

# ============================================================
# 3. Heatmap (hour x month)
# ============================================================
print("[*] Plot 3: Hour x Month heatmap...")
heatmap_data = hourly_clean.groupby([hourly_clean.index.month, hourly_clean.index.hour]).mean()
heatmap_data = heatmap_data.unstack(level=0)  # columns = months, index = hours

fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(heatmap_data.values, aspect="auto", cmap="YlOrRd",
               origin="lower", interpolation="nearest")
ax.set_xticks(range(12))
ax.set_xticklabels([calendar.month_abbr[m] for m in range(1, 13)])
ax.set_yticks(range(0, 24, 2))
ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
ax.set_xlabel("Month")
ax.set_ylabel("Hour of Day")
ax.set_title("Bardejov Heat Load - Average Power (MW) by Hour and Month")
cbar = fig.colorbar(im, ax=ax, pad=0.02)
cbar.set_label("MW")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "03_heatmap_hour_month.png", dpi=150)
print("[+] Saved 03_heatmap_hour_month.png")

# ============================================================
# 4. Weekday vs Weekend profiles
# ============================================================
print("[*] Plot 4: Weekday vs Weekend profiles...")
# Use only heating season (Oct-Mar) for meaningful comparison
heating = hourly_clean[hourly_clean.index.month.isin([10, 11, 12, 1, 2, 3])]
weekday = heating[heating.index.dayofweek < 5]
weekend = heating[heating.index.dayofweek >= 5]

wd_profile = weekday.groupby(weekday.index.hour).agg(["mean", "std"])
we_profile = weekend.groupby(weekend.index.hour).agg(["mean", "std"])

fig, ax = plt.subplots(figsize=(12, 6))
hours = range(24)

ax.plot(hours, wd_profile["mean"].values, linewidth=2, color="firebrick",
        label="Weekday", marker="o", markersize=3)
ax.fill_between(hours,
                wd_profile["mean"].values - wd_profile["std"].values,
                wd_profile["mean"].values + wd_profile["std"].values,
                alpha=0.15, color="firebrick")

ax.plot(hours, we_profile["mean"].values, linewidth=2, color="steelblue",
        label="Weekend", marker="s", markersize=3)
ax.fill_between(hours,
                we_profile["mean"].values - we_profile["std"].values,
                we_profile["mean"].values + we_profile["std"].values,
                alpha=0.15, color="steelblue")

ax.set_xlabel("Hour of Day")
ax.set_ylabel("Heat Load (MW)")
ax.set_title("Bardejov Heat Load - Weekday vs Weekend (Heating Season Oct-Mar)")
ax.set_xticks(range(0, 24, 2))
ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
ax.set_xlim(0, 23)
ax.set_ylim(0, None)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "04_weekday_vs_weekend.png", dpi=150)
print("[+] Saved 04_weekday_vs_weekend.png")

# ============================================================
# 5. Monthly energy totals (bar chart by year)
# ============================================================
print("[*] Plot 5: Monthly energy totals...")
daily_MWh = hourly_MW.resample("D").mean() * 24
monthly_MWh = daily_MWh.resample("MS").sum()
monthly_MWh = monthly_MWh[monthly_MWh > 0]  # drop zero months

pivot = monthly_MWh.groupby([monthly_MWh.index.year, monthly_MWh.index.month]).first().unstack(level=0)
pivot.index = [calendar.month_abbr[m] for m in pivot.index]

fig, ax = plt.subplots(figsize=(14, 6))
years = sorted(pivot.columns)
n_years = len(years)
width = 0.8 / n_years
x = np.arange(len(pivot.index))
colors_yr = plt.cm.viridis(np.linspace(0.2, 0.9, n_years))

for i, year in enumerate(years):
    vals = pivot[year].fillna(0).values
    ax.bar(x + i * width, vals, width, label=str(year), color=colors_yr[i], alpha=0.8)

ax.set_xticks(x + width * (n_years - 1) / 2)
ax.set_xticklabels(pivot.index)
ax.set_ylabel("Monthly Energy (MWh)")
ax.set_title("Bardejov Heat Load - Monthly Energy Totals by Year")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "05_monthly_energy_totals.png", dpi=150)
print("[+] Saved 05_monthly_energy_totals.png")

# ============================================================
# 6. Cumulative energy curve (normalized by year)
# ============================================================
print("[*] Plot 6: Cumulative energy curve...")
fig, ax = plt.subplots(figsize=(12, 6))
colors_cum = plt.cm.viridis(np.linspace(0.2, 0.9, 5))

for i, year in enumerate([2021, 2022, 2023, 2024, 2025]):
    year_daily = daily_MWh[daily_MWh.index.year == year]
    if len(year_daily) == 0:
        continue
    cumulative = year_daily.cumsum()
    doy = year_daily.index.dayofyear
    ax.plot(doy, cumulative.values / 1000, linewidth=1.5, label=f"{year} ({cumulative.iloc[-1]/1000:.0f} GWh)",
            color=colors_cum[i])

# Month labels on x-axis
month_starts = [pd.Timestamp(f"2024-{m:02d}-01").dayofyear for m in range(1, 13)]
ax.set_xticks(month_starts)
ax.set_xticklabels([calendar.month_abbr[m] for m in range(1, 13)])
ax.set_xlabel("")
ax.set_ylabel("Cumulative Energy (GWh)")
ax.set_title("Bardejov Heat Load - Cumulative Annual Energy")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "06_cumulative_energy.png", dpi=150)
print("[+] Saved 06_cumulative_energy.png")

# ============================================================
# 7. One-week zoom: Winter
# ============================================================
print("[*] Plot 7: Winter week zoom...")
winter_start, winter_end = "2024-01-15", "2024-01-22"
winter_week = hourly_MW[winter_start:winter_end]

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(winter_week.index, winter_week.values, linewidth=1, color="firebrick", marker=".", markersize=2)
ax.fill_between(winter_week.index, winter_week.values, alpha=0.15, color="firebrick")
ax.set_ylabel("Heat Load (MW)")
ax.set_title(f"Bardejov Heat Load - Winter Week ({winter_start} to {winter_end})")
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b"))
ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
ax.tick_params(axis="x", which="major", labelsize=8)
ax.tick_params(axis="x", which="minor", length=3)
ax.set_ylim(0, None)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "07_winter_week.png", dpi=150)
print("[+] Saved 07_winter_week.png")

# ============================================================
# 8. One-week zoom: Summer
# ============================================================
print("[*] Plot 8: Summer week zoom...")
summer_start, summer_end = "2024-07-15", "2024-07-22"
summer_week = hourly_MW[summer_start:summer_end]

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(summer_week.index, summer_week.values, linewidth=1, color="steelblue", marker=".", markersize=2)
ax.fill_between(summer_week.index, summer_week.values, alpha=0.15, color="steelblue")
ax.set_ylabel("Heat Load (MW)")
ax.set_title(f"Bardejov Heat Load - Summer Week ({summer_start} to {summer_end})")
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b"))
ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
ax.tick_params(axis="x", which="major", labelsize=8)
ax.tick_params(axis="x", which="minor", length=3)
ax.set_ylim(0, None)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "08_summer_week.png", dpi=150)
print("[+] Saved 08_summer_week.png")

plt.close("all")
print("[+] All done.")
