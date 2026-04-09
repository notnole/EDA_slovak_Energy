"""Plot 3 representative daily heat load profiles overlaid."""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
DATA_FILE = Path(__file__).resolve().parents[4] / "data" / "Bardejov" / "heat_load_timeseries.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1]

# Load and resample to hourly MW
print("[*] Loading heat load data...")
df = pd.read_csv(DATA_FILE, parse_dates=["datetime"])
df = df.set_index("datetime").sort_index()
hourly_MW = df["heat_load_kW"].resample("h").mean() / 1000

# Representative days
days = {
    "Winter peak (2022-01-18)": "2022-01-18",
    "Shoulder season (2023-10-20)": "2023-10-20",
    "Summer baseload (2023-07-11)": "2023-07-11",
}
colors = ["firebrick", "darkorange", "steelblue"]

fig, ax = plt.subplots(figsize=(12, 6))

for (label, date), color in zip(days.items(), colors):
    day = hourly_MW[date]
    hours = range(len(day))
    daily_MWh = day.mean() * 24
    ax.plot(hours, day.values, linewidth=2, label=f"{label} [{daily_MWh:.0f} MWh]",
            color=color, marker="o", markersize=3)

ax.set_xlabel("Hour of Day")
ax.set_ylabel("Heat Load (MW)")
ax.set_title("Bardejov District Heating - Representative Daily Profiles")
ax.set_xticks(range(0, 24, 2))
ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
ax.set_xlim(0, 23)
ax.set_ylim(0, None)
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "04_daily_profiles.png", dpi=150)
print("[+] Saved 04_daily_profiles.png")

plt.close("all")
print("[+] Done.")
