"""Temperature sensitivity by hour and residual analysis after temperature fit."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
}).dropna(subset=["heat_MW", "temp"])

daily = merged.resample("D").mean().dropna()
print(f"[+] {len(merged):,} hourly, {len(daily)} daily records")

# ============================================================
# 4. Temperature sensitivity by hour of day
# ============================================================
print("[*] Plot 4: Temperature sensitivity by hour...")
heating_hours = merged[merged["temp"] <= 15].copy()
slopes_by_hour = []
r_by_hour = []

for h in range(24):
    hour_data = heating_hours[heating_hours.index.hour == h]
    if len(hour_data) > 30:
        s, i, r, p, se = stats.linregress(hour_data["temp"], hour_data["heat_MW"])
        slopes_by_hour.append(s)
        r_by_hour.append(r)
    else:
        slopes_by_hour.append(np.nan)
        r_by_hour.append(np.nan)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1.bar(range(24), slopes_by_hour, color="firebrick", alpha=0.7)
ax1.set_ylabel("Slope (MW / degC)")
ax1.set_title("Temperature Sensitivity by Hour of Day (heating season, T <= 15 degC)")
ax1.axhline(0, color="black", linewidth=0.5)
ax1.grid(True, alpha=0.3, axis="y")

ax2.bar(range(24), r_by_hour, color="steelblue", alpha=0.7)
ax2.set_ylabel("Pearson r")
ax2.set_xlabel("Hour of Day")
ax2.set_xticks(range(0, 24, 2))
ax2.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
ax2.axhline(0, color="black", linewidth=0.5)
ax2.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "04_temp_sensitivity_by_hour.png", dpi=150)
print("[+] Saved 04_temp_sensitivity_by_hour.png")

# ============================================================
# 5. Residuals after temperature fit
# ============================================================
print("[*] Plot 5: Residuals after temperature fit...")
daily_heating = daily[daily["temp"] <= 15].copy()
slope_d, intercept_d, r_d, _, _ = stats.linregress(daily_heating["temp"], daily_heating["heat_MW"])

daily["predicted"] = np.where(daily["temp"] <= 15,
                               slope_d * daily["temp"] + intercept_d,
                               daily["heat_MW"].median())
daily["residual"] = daily["heat_MW"] - daily["predicted"]

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

ax = axes[0]
ax.plot(daily.index, daily["residual"], linewidth=0.8, color="gray")
ax.axhline(0, color="black", linewidth=0.5)
ax.set_ylabel("Residual (MW)")
ax.set_title(f"Residual after Temperature Fit (daily, heating r={r_d:.2f})")
ax.grid(True, alpha=0.3)

ax = axes[1]
dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
residual_by_dow = [daily_heating.merge(daily[["residual"]], left_index=True, right_index=True)
                   .loc[lambda x: x.index.dayofweek == d, "residual"].values for d in range(7)]
bp = ax.boxplot(residual_by_dow, patch_artist=True, whis=(10, 90))
for i, patch in enumerate(bp["boxes"]):
    patch.set_facecolor("steelblue" if i < 5 else "darkorange")
    patch.set_alpha(0.6)
ax.set_xticklabels(dow_names)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_ylabel("Residual (MW)")
ax.set_title("Temperature Residual by Day of Week (heating season)")
ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "05_residuals.png", dpi=150)
print("[+] Saved 05_residuals.png")

plt.close("all")
print("[+] Done.")
