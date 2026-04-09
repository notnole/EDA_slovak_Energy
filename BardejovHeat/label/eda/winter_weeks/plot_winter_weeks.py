"""Plot 5 representative winter weeks from 3 different winters."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Paths
DATA_FILE = Path(__file__).resolve().parents[4] / "data" / "Bardejov" / "heat_load_timeseries.csv"
OUTPUT_DIR = Path(__file__).resolve().parent

# Load and resample to hourly MW
print("[*] Loading heat load data...")
df = pd.read_csv(DATA_FILE, parse_dates=["datetime"])
df = df.set_index("datetime").sort_index()
hourly_MW = df["heat_load_kW"].resample("h").mean() / 1000

# 5 weeks across 3 winters (early, mid, late winter mix)
weeks = [
    ("2022-01-10", "2022-01-17", "Winter 2021/22 - Jan cold spell"),
    ("2022-12-05", "2022-12-12", "Winter 2022/23 - Early Dec"),
    ("2023-02-06", "2023-02-13", "Winter 2022/23 - Feb"),
    ("2024-01-08", "2024-01-15", "Winter 2023/24 - Jan peak"),
    ("2024-11-18", "2024-11-25", "Winter 2024/25 - Nov shoulder"),
]

for i, (start, end, title) in enumerate(weeks, 1):
    week = hourly_MW[start:end]
    if len(week) == 0:
        print(f"[-] No data for {start} to {end}, skipping")
        continue

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(week.index, week.values, linewidth=1, color="firebrick", marker=".", markersize=2)
    ax.fill_between(week.index, week.values, alpha=0.15, color="firebrick")

    # Stats annotation
    peak = week.max()
    mean = week.mean()
    daily_MWh = week.resample("D").mean() * 24
    ax.text(0.98, 0.95, f"Peak: {peak:.1f} MW | Mean: {mean:.1f} MW | ~{daily_MWh.mean():.0f} MWh/day",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_ylabel("Heat Load (MW)")
    ax.set_title(f"Bardejov Heat Load - {title}")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
    ax.tick_params(axis="x", which="major", labelsize=8)
    ax.tick_params(axis="x", which="minor", length=3)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fname = f"{i:02d}_{start}_to_{end}.png"
    fig.savefig(OUTPUT_DIR / fname, dpi=150)
    print(f"[+] Saved {fname}")

plt.close("all")
print("[+] All done.")
