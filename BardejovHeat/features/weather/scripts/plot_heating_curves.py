"""Heating curve plots: temperature vs heat load relationships."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy import stats

# Paths
DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "Bardejov"
HEAT_FILE = DATA_DIR / "heat_load_timeseries.csv"
WEATHER_FILE = DATA_DIR / "Weather" / "bardejov_weather_actual.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1]

# Load heat load -> hourly MW
print("[*] Loading heat load data...")
df_heat = pd.read_csv(HEAT_FILE, parse_dates=["datetime"])
df_heat = df_heat.set_index("datetime").sort_index()
hourly_MW = df_heat["heat_load_kW"].resample("h").mean() / 1000

# Load weather
print("[*] Loading weather data...")
df_wx = pd.read_csv(WEATHER_FILE, parse_dates=["time"])
df_wx = df_wx.set_index("time").sort_index()

# Merge on overlapping period
merged = pd.DataFrame({
    "heat_MW": hourly_MW,
    "temp": df_wx["temperature_2m"],
    "apparent_temp": df_wx["apparent_temperature"],
}).dropna(subset=["heat_MW", "temp"])

daily = merged.resample("D").mean().dropna()
print(f"[+] {len(merged):,} hourly, {len(daily)} daily records")

# ============================================================
# 1. Heating curve: Heat load vs Temperature (scatter + fit)
# ============================================================
print("[*] Plot 1: Heating curve...")
fig, ax = plt.subplots(figsize=(12, 7))

hours = merged.index.hour
scatter = ax.scatter(merged["temp"], merged["heat_MW"], c=hours, cmap="twilight",
                     s=1, alpha=0.3, rasterized=True)
cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
cbar.set_label("Hour of Day")

ax.scatter(daily["temp"], daily["heat_MW"], color="black", s=8, alpha=0.6,
           zorder=5, label="Daily mean")

threshold = 15
heating = daily[daily["temp"] <= threshold]
non_heating = daily[daily["temp"] > threshold]

if len(heating) > 10:
    slope, intercept, r, p, se = stats.linregress(heating["temp"], heating["heat_MW"])
    x_fit = np.linspace(heating["temp"].min(), threshold, 100)
    ax.plot(x_fit, slope * x_fit + intercept, "r-", linewidth=2,
            label=f"Heating: {slope:.2f} MW/degC (r={r:.2f})")

if len(non_heating) > 10:
    slope_nh, intercept_nh, r_nh, p_nh, se_nh = stats.linregress(non_heating["temp"], non_heating["heat_MW"])
    x_fit_nh = np.linspace(threshold, non_heating["temp"].max(), 100)
    ax.plot(x_fit_nh, slope_nh * x_fit_nh + intercept_nh, "b-", linewidth=2,
            label=f"Non-heating: {slope_nh:.2f} MW/degC (r={r_nh:.2f})")

ax.axvline(threshold, color="gray", linestyle="--", alpha=0.5, label=f"Threshold {threshold} degC")
ax.set_xlabel("Outdoor Temperature (degC)")
ax.set_ylabel("Heat Load (MW)")
ax.set_title("Bardejov Heating Curve - Heat Load vs Temperature")
ax.legend(loc="upper right")
ax.set_ylim(0, None)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "01_heating_curve.png", dpi=150)
print("[+] Saved 01_heating_curve.png")

# ============================================================
# 2. Daily heating curve by month
# ============================================================
print("[*] Plot 2: Daily heating curve by month...")
fig, ax = plt.subplots(figsize=(12, 7))

months = daily.index.month
month_cmap = plt.cm.hsv(np.linspace(0, 0.85, 12))
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

for m in range(1, 13):
    mask = months == m
    if mask.sum() > 0:
        ax.scatter(daily.loc[mask, "temp"], daily.loc[mask, "heat_MW"],
                   s=15, alpha=0.7, color=month_cmap[m-1], label=month_names[m-1])

ax.set_xlabel("Daily Mean Temperature (degC)")
ax.set_ylabel("Daily Mean Heat Load (MW)")
ax.set_title("Bardejov Heating Curve - Daily Means by Month")
ax.legend(ncol=4, fontsize=8, loc="upper right")
ax.set_ylim(0, None)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "02_heating_curve_monthly.png", dpi=150)
print("[+] Saved 02_heating_curve_monthly.png")

# ============================================================
# 2b. Time series: daily MWh and daily avg temperature
# ============================================================
print("[*] Plot 2b: Daily MWh + temperature time series...")
daily_MWh = daily.copy()
daily_MWh["heat_MWh"] = daily_MWh["heat_MW"] * 24

fig, ax1 = plt.subplots(figsize=(16, 6))

ax1.fill_between(daily_MWh.index, daily_MWh["heat_MWh"], alpha=0.3, color="firebrick")
ax1.plot(daily_MWh.index, daily_MWh["heat_MWh"], linewidth=0.8, color="firebrick", label="Heat (MWh/day)")
ax1.set_ylabel("Daily Heat Production (MWh)", color="firebrick")
ax1.tick_params(axis="y", labelcolor="firebrick")
ax1.set_ylim(0, None)

ax2 = ax1.twinx()
ax2.plot(daily_MWh.index, daily_MWh["temp"], linewidth=0.8, color="steelblue", alpha=0.8, label="Temperature")
ax2.set_ylabel("Daily Mean Temperature (degC)", color="steelblue")
ax2.tick_params(axis="y", labelcolor="steelblue")
ax2.invert_yaxis()
ax2.axhline(0, color="steelblue", linestyle=":", alpha=0.3)

ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax1.set_title("Bardejov - Daily Heat Production and Temperature (2024-2025)")
ax1.grid(True, alpha=0.3)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "02b_daily_temp_vs_MWh.png", dpi=150)
print("[+] Saved 02b_daily_temp_vs_MWh.png")

# ============================================================
# 6. Apparent vs actual temperature
# ============================================================
print("[*] Plot 6: Apparent vs actual temperature effect...")
fig, ax = plt.subplots(figsize=(12, 7))

ax.scatter(daily["temp"], daily["heat_MW"], s=15, alpha=0.5, color="steelblue", label="Actual temp")
ax.scatter(daily["apparent_temp"], daily["heat_MW"], s=15, alpha=0.5, color="firebrick", label="Apparent temp")

for col, color, label in [("temp", "steelblue", "Actual"), ("apparent_temp", "firebrick", "Apparent")]:
    d = daily[[col, "heat_MW"]].dropna()
    d_heat = d[d[col] <= 15]
    if len(d_heat) > 10:
        s, i, r, p, se = stats.linregress(d_heat[col], d_heat["heat_MW"])
        x_f = np.linspace(d_heat[col].min(), 15, 100)
        ax.plot(x_f, s * x_f + i, color=color, linewidth=2,
                label=f"{label}: {s:.2f} MW/degC, r={r:.2f}")

ax.set_xlabel("Temperature (degC)")
ax.set_ylabel("Daily Mean Heat Load (MW)")
ax.set_title("Actual vs Apparent Temperature - Which Predicts Better?")
ax.legend()
ax.set_ylim(0, None)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "06_apparent_vs_actual_temp.png", dpi=150)
print("[+] Saved 06_apparent_vs_actual_temp.png")

plt.close("all")
print("[+] Done.")
