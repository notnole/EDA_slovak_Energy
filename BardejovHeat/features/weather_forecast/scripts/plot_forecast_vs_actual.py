"""Scatter plots: forecast temperature vs actual temperature by model and lead time."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Paths
DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "Bardejov" / "Weather"
ACTUAL_FILE = DATA_DIR / "bardejov_weather_actual.csv"
DA_FORECAST_FILE = DATA_DIR / "bardejov_da_forecasts.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1]

# Load data
print("[*] Loading weather data...")
actual = pd.read_csv(ACTUAL_FILE, parse_dates=["time"]).set_index("time")
da_fc = pd.read_csv(DA_FORECAST_FILE, parse_dates=["time"]).set_index("time")

# Models and lead times to plot
models = {
    "Best Match - Now":  "best_match_temp_now",
    "Best Match - DA+1": "best_match_temp_da1",
    "Best Match - DA+2": "best_match_temp_da2",
    "GFS - Now":         "gfs_seamless_temp_now",
    "GFS - DA+1":        "gfs_seamless_temp_da1",
    "GFS - DA+2":        "gfs_seamless_temp_da2",
}

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for ax, (name, col) in zip(axes.flat, models.items()):
    merged = pd.DataFrame({
        "actual": actual["temperature_2m"],
        "forecast": da_fc[col],
    }).dropna()

    n = len(merged)
    mae = np.abs(merged["actual"] - merged["forecast"]).mean()
    rmse = np.sqrt(((merged["actual"] - merged["forecast"]) ** 2).mean())
    slope, intercept, r, _, _ = stats.linregress(merged["actual"], merged["forecast"])
    bias = (merged["forecast"] - merged["actual"]).mean()

    ax.scatter(merged["actual"], merged["forecast"], s=1, alpha=0.1, color="steelblue", rasterized=True)

    # Perfect line
    lims = [min(merged["actual"].min(), merged["forecast"].min()) - 1,
            max(merged["actual"].max(), merged["forecast"].max()) + 1]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="Perfect")

    # Fit line
    x_fit = np.linspace(lims[0], lims[1], 100)
    ax.plot(x_fit, slope * x_fit + intercept, "r-", linewidth=1.5, alpha=0.7,
            label=f"Fit: y={slope:.2f}x{intercept:+.1f}")

    ax.set_xlabel("Actual Temperature (degC)")
    ax.set_ylabel("Forecast Temperature (degC)")
    ax.set_title(name)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Stats box
    stats_text = f"n={n:,}\nr={r:.3f}\nMAE={mae:.2f} degC\nRMSE={rmse:.2f} degC\nBias={bias:+.2f} degC"
    ax.text(0.97, 0.03, stats_text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            family="monospace")

fig.suptitle("Bardejov Temperature Forecast vs Actual - By Model and Lead Time", fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "01_forecast_vs_actual_temp.png", dpi=150)
print("[+] Saved 01_forecast_vs_actual_temp.png")

plt.close("all")
print("[+] Done.")
