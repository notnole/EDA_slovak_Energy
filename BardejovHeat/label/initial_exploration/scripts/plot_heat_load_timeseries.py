"""Plot heat load time series for Bardejov district heating."""

import calendar
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Paths
DATA_FILE = Path(__file__).resolve().parents[4] / "data" / "Bardejov" / "heat_load_timeseries.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1]

# Load data
print("[*] Loading heat load data...")
df = pd.read_csv(DATA_FILE, parse_dates=["datetime"])
df = df.set_index("datetime").sort_index()
print(f"[+] Loaded {len(df):,} rows: {df.index[0]} to {df.index[-1]}")
# Values are power in kW. CSV already corrected for the 2025 kWh/15min issue.
print(f"[+] Heat load range: {df['heat_load_kW'].min():,.0f} - {df['heat_load_kW'].max():,.0f} kW")

# Remove obvious data errors (> 30 MW, normal peaks ~23 MW)
outlier_mask = df["heat_load_kW"] > 30_000
n_outliers = outlier_mask.sum()
if n_outliers > 0:
    print(f"[!] Removing {n_outliers} outlier(s) > 30,000 kW (data errors)")
    df.loc[outlier_mask, "heat_load_kW"] = pd.NA
    df["heat_load_kW"] = df["heat_load_kW"].interpolate(method="time")

# Normalize mixed resolution (hourly Jan-Apr, 15-min May-Dec).
# All values in kW. Resample to hourly mean for consistent resolution.
hourly_kW = df["heat_load_kW"].resample("h").mean()
hourly_MW = hourly_kW / 1000

# --- Plot 1: Full time series - power (MW) ---
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(hourly_MW.index, hourly_MW.values, linewidth=0.3, alpha=0.7, color="firebrick")
ax.set_ylabel("Heat Load (MW)")
ax.set_title("Bardejov District Heating - Heat Load Power (2021-2025)")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.xaxis.set_minor_formatter(mdates.DateFormatter("%m"))
ax.tick_params(axis="x", which="major", labelsize=7)
ax.tick_params(axis="x", which="minor", labelsize=7)
ax.set_xlim(hourly_MW.index[0], hourly_MW.index[-1])
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "01_heat_load_full_timeseries.png", dpi=150)
print("[+] Saved 01_heat_load_full_timeseries.png")

# --- Plot 2: Daily energy total (mean_MW * 24h = MWh/day) ---
daily = hourly_MW.resample("D").mean() * 24

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(daily.index, daily.values, linewidth=0.8, color="firebrick")
ax.set_ylabel("Daily Heat Load (MWh)")
ax.set_title("Bardejov District Heating - Daily Total Heat Load")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.xaxis.set_minor_formatter(mdates.DateFormatter("%m"))
ax.tick_params(axis="x", which="major", labelsize=7)
ax.tick_params(axis="x", which="minor", labelsize=7)
ax.set_xlim(daily.index[0], daily.index[-1])
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "02_daily_heat_load.png", dpi=150)
print("[+] Saved 02_daily_heat_load.png")

# --- Plot 3: Year-over-year overlay by month ---
fig, ax = plt.subplots(figsize=(14, 6))
for year in sorted(df.index.year.unique()):
    subset = daily[daily.index.year == year]
    day_of_year = subset.index.dayofyear
    ax.plot(day_of_year, subset.values, linewidth=0.8, label=str(year), alpha=0.8)

month_starts = [pd.Timestamp(f"2024-{m:02d}-01").dayofyear for m in range(1, 13)]
month_labels = [calendar.month_abbr[m] for m in range(1, 13)]
ax.set_xticks(month_starts)
ax.set_xticklabels(month_labels)
ax.set_xlabel("")
ax.set_ylabel("Daily Heat Load (MWh)")
ax.set_title("Bardejov Heat Load - Year-over-Year Comparison")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "03_year_over_year.png", dpi=150)
print("[+] Saved 03_year_over_year.png")

plt.close("all")
print("[+] Done.")
