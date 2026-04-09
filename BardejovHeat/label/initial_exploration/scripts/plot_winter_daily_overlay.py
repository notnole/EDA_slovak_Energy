"""Plot all winter days overlaid on a single 24h axis, colored by date."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from pathlib import Path

# Paths
DATA_FILE = Path(__file__).resolve().parents[4] / "data" / "Bardejov" / "heat_load_timeseries.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1]

# Load and resample to hourly MW
print("[*] Loading heat load data...")
df = pd.read_csv(DATA_FILE, parse_dates=["datetime"])
df = df.set_index("datetime").sort_index()
hourly_MW = df["heat_load_kW"].resample("h").mean() / 1000

# Define winter months (Oct-Mar heating season)
winter_mask = hourly_MW.index.month.isin([10, 11, 12, 1, 2, 3])
winter = hourly_MW[winter_mask]

# Group by date
daily_groups = winter.groupby(winter.index.date)

# Build list of (date, hourly_values) with exactly 24 hours
days = []
for date, group in daily_groups:
    if len(group) == 24:
        days.append((pd.Timestamp(date), group.values))

days.sort(key=lambda x: x[0])
n_days = len(days)
print(f"[+] {n_days} complete winter days")

# Colormap: yellow (oldest) -> green (most recent)
cmap = mcolors.LinearSegmentedColormap.from_list("yellow_green", ["#FF0000", "#BFFF00"])
norm = plt.Normalize(0, n_days - 1)

fig, ax = plt.subplots(figsize=(14, 7))

for i, (date, values) in enumerate(days):
    ax.plot(range(24), values, linewidth=0.3, alpha=0.4, color=cmap(norm(i)))

ax.set_xlabel("Hour of Day")
ax.set_ylabel("Heat Load (MW)")
ax.set_title("Bardejov District Heating - Winter Daily Profiles (Oct-Mar, 2021-2025)")
ax.set_xticks(range(0, 24, 2))
ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
ax.set_xlim(0, 23)
ax.set_ylim(0, None)
ax.grid(True, alpha=0.3)

# Colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
first_date = days[0][0].strftime("%b %Y")
last_date = days[-1][0].strftime("%b %Y")
cbar.set_ticks([0, 1])
cbar.set_ticklabels([first_date, last_date])

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "05_winter_daily_overlay.png", dpi=150)
print("[+] Saved 05_winter_daily_overlay.png")

plt.close("all")
print("[+] Done.")
