"""Forecast accuracy for all weather variables (short-range forecasts)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Paths
DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "Bardejov" / "Weather"
ACTUAL_FILE = DATA_DIR / "bardejov_weather_actual.csv"
FORECAST_FILE = DATA_DIR / "bardejov_weather_forecasts.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1]

# Load data
print("[*] Loading weather data...")
actual = pd.read_csv(ACTUAL_FILE, parse_dates=["time"]).set_index("time")
forecast = pd.read_csv(FORECAST_FILE, parse_dates=["time"]).set_index("time")

# Variable pairs: (display name, actual col, forecast col, unit)
variables = [
    ("Temperature",       "temperature_2m",       "best_match_temperature_2m",       "degC"),
    ("Humidity",          "relative_humidity_2m",  "best_match_relative_humidity_2m", "%"),
    ("Wind Speed",        "windspeed_10m",         "best_match_wind_speed_10m",       "km/h"),
    ("Solar Radiation",   "shortwave_radiation",   "best_match_shortwave_radiation",  "W/m2"),
    ("Cloud Cover",       "cloudcover",            "best_match_cloud_cover",          "%"),
    ("Precipitation",     "precipitation",         "best_match_precipitation",        "mm"),
    ("Snow Depth",        "snow_depth",            "best_match_snow_depth",           "m"),
    ("Surface Pressure",  "surface_pressure",      "best_match_surface_pressure",     "hPa"),
]

fig, axes = plt.subplots(2, 4, figsize=(22, 11))

for ax, (name, act_col, fc_col, unit) in zip(axes.flat, variables):
    merged = pd.DataFrame({
        "actual": actual[act_col],
        "forecast": forecast[fc_col],
    }).dropna()

    n = len(merged)
    if n < 100:
        ax.text(0.5, 0.5, f"{name}\nInsufficient data", transform=ax.transAxes,
                ha="center", va="center")
        continue

    mae = np.abs(merged["actual"] - merged["forecast"]).mean()
    rmse = np.sqrt(((merged["actual"] - merged["forecast"]) ** 2).mean())
    r_val = merged["actual"].corr(merged["forecast"])
    bias = (merged["forecast"] - merged["actual"]).mean()

    ax.scatter(merged["actual"], merged["forecast"], s=1, alpha=0.08, color="steelblue", rasterized=True)

    lims = [min(merged["actual"].min(), merged["forecast"].min()),
            max(merged["actual"].max(), merged["forecast"].max())]
    margin = (lims[1] - lims[0]) * 0.05
    lims = [lims[0] - margin, lims[1] + margin]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5)

    ax.set_xlabel(f"Actual ({unit})")
    ax.set_ylabel(f"Forecast ({unit})")
    ax.set_title(name)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    stats_text = f"n={n:,}\nr={r_val:.3f}\nMAE={mae:.2f} {unit}\nBias={bias:+.2f}"
    ax.text(0.97, 0.03, stats_text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            family="monospace")

fig.suptitle("Bardejov Weather Forecast Accuracy - All Variables (Best Match, short-range)", fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "02_all_variables_forecast_accuracy.png", dpi=150)
print("[+] Saved 02_all_variables_forecast_accuracy.png")

plt.close("all")
print("[+] Done.")
