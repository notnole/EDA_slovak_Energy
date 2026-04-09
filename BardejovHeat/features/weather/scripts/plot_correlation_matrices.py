"""Correlation matrices: weather features vs heat load, split by day/night and season."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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
    "humidity": df_wx["relative_humidity_2m"],
    "wind": df_wx["windspeed_10m"],
    "solar": df_wx["shortwave_radiation"],
    "cloud": df_wx["cloudcover"],
    "snow_depth": df_wx["snow_depth"],
}).dropna(subset=["heat_MW", "temp"])

print(f"[+] {len(merged):,} overlapping hourly records")

# Common settings
corr_cols = ["heat_MW", "temp", "apparent_temp", "humidity", "wind", "solar", "cloud", "snow_depth"]
labels = ["Heat Load", "Temperature", "Apparent Temp", "Humidity", "Wind", "Solar", "Cloud", "Snow Depth"]


def plot_corr_matrix(ax, subset, title):
    """Plot a single correlation matrix on the given axes."""
    corr = subset[corr_cols].corr()
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_cols)))
    ax.set_yticks(range(len(corr_cols)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)
    ax.set_title(title, fontsize=11)
    return im


# ============================================================
# 3. Day vs Night
# ============================================================
print("[*] Plot 3: Day/Night correlation matrices...")
night = merged[merged.index.hour.isin(list(range(0, 7)) + [22, 23])]
day = merged[merged.index.hour.isin(range(7, 22))]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

for ax, subset, title in [(ax1, day, f"Day (07-22h, n={len(day):,})"),
                           (ax2, night, f"Night (22-07h, n={len(night):,})")]:
    im = plot_corr_matrix(ax, subset, title)
    fig.colorbar(im, ax=ax, pad=0.02, shrink=0.8).set_label("Pearson r")

fig.suptitle("Weather Features - Correlation Matrix: Day vs Night", fontsize=13)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "03_correlation_matrix_day_night.png", dpi=150)
print("[+] Saved 03_correlation_matrix_day_night.png")

# ============================================================
# 3b. Season x Day/Night (3x2 grid)
# ============================================================
print("[*] Plot 3b: Season x Day/Night correlation matrices...")
winter_mask = merged.index.month.isin([12, 1, 2])
shoulder_mask = merged.index.month.isin([3, 4, 10, 11])
summer_mask = merged.index.month.isin([5, 6, 7, 8, 9])

day_mask = merged.index.hour.isin(range(7, 22))
night_mask = ~day_mask

slices = {
    "Winter Day":     merged[winter_mask & day_mask],
    "Winter Night":   merged[winter_mask & night_mask],
    "Shoulder Day":   merged[shoulder_mask & day_mask],
    "Shoulder Night": merged[shoulder_mask & night_mask],
    "Summer Day":     merged[summer_mask & day_mask],
    "Summer Night":   merged[summer_mask & night_mask],
}

fig, axes = plt.subplots(3, 2, figsize=(18, 24))

for ax, (title, subset) in zip(axes.flat, slices.items()):
    im = plot_corr_matrix(ax, subset, f"{title} (n={len(subset):,})")
    fig.colorbar(im, ax=ax, pad=0.02, shrink=0.8).set_label("Pearson r")

fig.suptitle("Weather Features - Correlation Matrix: Season x Day/Night", fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "03b_correlation_matrix_season_daynight.png", dpi=150)
print("[+] Saved 03b_correlation_matrix_season_daynight.png")

plt.close("all")
print("[+] Done.")
