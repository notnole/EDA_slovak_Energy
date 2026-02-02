"""Quick check if IDM arbitrage disappeared after Dec 15."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
MASTER_PATH = BASE_DIR / "data" / "master" / "master_imbalance_data.csv"
IDM_PATH = BASE_DIR / "RawData" / "IDM_MarketData"
OUTPUT_DIR = Path(__file__).parent

# Load IDM
print("[*] Loading IDM data...")
all_data = []
for folder in IDM_PATH.iterdir():
    if folder.is_dir() and folder.name.startswith("IDM_total"):
        csv_path = folder / "15 min.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, sep=";", decimal=",")
            all_data.append(df)
            print(f"  [+] {folder.name}")

idm = pd.concat(all_data, ignore_index=True)
idm["date"] = pd.to_datetime(idm["Delivery day"], format="%d.%m.%Y")
idm["period_num"] = idm["Period number"]
idm["hour"] = (idm["period_num"] - 1) // 4
idm["qh_in_hour"] = ((idm["period_num"] - 1) % 4) + 1
idm["datetime"] = idm["date"] + pd.to_timedelta(idm["hour"], unit="h") + pd.to_timedelta((idm["qh_in_hour"]-1)*15, unit="m")
idm["idm_price"] = pd.to_numeric(idm["Weighted average price of all trades (EUR/MWh)"], errors="coerce")
idm = idm[idm["qh_in_hour"].isin([1, 2])].copy()

# Load imbalance
print("[*] Loading imbalance data...")
imb = pd.read_csv(MASTER_PATH, parse_dates=["datetime"])
imb = imb[["datetime", "System Imbalance (MWh)", "Imbalance Settlement Price (EUR/MWh)"]].copy()
imb.columns = ["datetime", "imbalance", "imb_price"]
imb = imb[imb["imbalance"].abs() <= 300]

print(f"  IDM range: {idm['date'].min()} to {idm['date'].max()}")
print(f"  Imbalance range: {imb['datetime'].min()} to {imb['datetime'].max()}")

# Merge
merged = pd.merge(idm[["datetime", "hour", "idm_price"]], imb, on="datetime", how="inner")
merged = merged.dropna()
merged["year"] = merged["datetime"].dt.year
merged["month"] = merged["datetime"].dt.month
merged["day"] = merged["datetime"].dt.day
merged = merged[(merged["hour"] >= 5) & (merged["hour"] < 22)]

# Filter to Sep 2025 onwards
merged = merged[(merged["year"] >= 2025) & ((merged["year"] > 2025) | (merged["month"] >= 9))]
merged = merged.sort_values("datetime").reset_index(drop=True)

# Calculate spread
merged["spread"] = merged["idm_price"] - merged["imb_price"]

print(f"\n[+] Total samples: {len(merged):,}")

# Define periods
def get_period(row):
    if row["year"] == 2025:
        if row["month"] < 12:
            return "Sep-Nov 2025"
        elif row["day"] <= 14:
            return "Dec 1-14"
        else:
            return "Dec 15-31"
    else:
        return "Jan 2026"

merged["period"] = merged.apply(get_period, axis=1)

# Print statistics by period
print("\n" + "=" * 80)
print("IDM-Imbalance Spread by Period")
print("=" * 80)
print(f"{'Period':<15} {'N':>8} {'Avg Spread':>12} {'Median':>10} {'Win Rate':>10} {'Total':>12}")
print("-" * 80)

for period in ["Sep-Nov 2025", "Dec 1-14", "Dec 15-31", "Jan 2026"]:
    subset = merged[merged["period"] == period]
    if len(subset) < 10:
        continue
    print(f"{period:<15} {len(subset):>8} {subset['spread'].mean():>10.1f}   {subset['spread'].median():>10.1f} {(subset['spread']>0).mean()*100:>8.0f}%   {subset['spread'].sum():>10.0f}")

print("-" * 80)
print(f"{'TOTAL':<15} {len(merged):>8} {merged['spread'].mean():>10.1f}   {merged['spread'].median():>10.1f} {(merged['spread']>0).mean()*100:>8.0f}%   {merged['spread'].sum():>10.0f}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Cumulative profit over time with period markers
ax1 = axes[0, 0]
merged["cum_profit"] = merged["spread"].cumsum()
ax1.plot(merged["datetime"], merged["cum_profit"], "g-", linewidth=1.5)
ax1.axhline(y=0, color="black", linewidth=0.5)

# Add vertical lines for period boundaries
for date_str, label in [("2025-12-01", "Dec 1"), ("2025-12-15", "Dec 15"), ("2026-01-01", "Jan 1")]:
    date = pd.to_datetime(date_str)
    if date >= merged["datetime"].min() and date <= merged["datetime"].max():
        ax1.axvline(x=date, color="red", linestyle="--", alpha=0.7)
        ax1.text(date, ax1.get_ylim()[1]*0.95, label, rotation=90, va="top", ha="right", fontsize=9)

ax1.set_xlabel("Date")
ax1.set_ylabel("Cumulative Profit (EUR/MWh)")
ax1.set_title("Cumulative P&L: Always Sell IDM Strategy")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

# Plot 2: Daily average spread
ax2 = axes[0, 1]
daily = merged.groupby(merged["datetime"].dt.date).agg({"spread": "mean"}).reset_index()
daily["datetime"] = pd.to_datetime(daily["datetime"])
colors = ["green" if s > 0 else "red" for s in daily["spread"]]
ax2.bar(daily["datetime"], daily["spread"], color=colors, alpha=0.7, width=0.8)
ax2.axhline(y=0, color="black", linewidth=1)

for date_str in ["2025-12-01", "2025-12-15", "2026-01-01"]:
    date = pd.to_datetime(date_str)
    if date >= daily["datetime"].min() and date <= daily["datetime"].max():
        ax2.axvline(x=date, color="blue", linestyle="--", alpha=0.7)

ax2.set_xlabel("Date")
ax2.set_ylabel("Daily Avg Spread (EUR/MWh)")
ax2.set_title("Daily Average IDM-Imbalance Spread")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

# Plot 3: Period comparison bars
ax3 = axes[1, 0]
periods = ["Sep-Nov 2025", "Dec 1-14", "Dec 15-31", "Jan 2026"]
avgs = []
win_rates = []
for p in periods:
    subset = merged[merged["period"] == p]
    if len(subset) >= 10:
        avgs.append(subset["spread"].mean())
        win_rates.append((subset["spread"] > 0).mean() * 100)
    else:
        avgs.append(0)
        win_rates.append(0)

x = np.arange(len(periods))
colors = ["green" if a > 5 else "orange" if a > 0 else "red" for a in avgs]
bars = ax3.bar(x, avgs, color=colors, edgecolor="black", alpha=0.8)
ax3.axhline(y=0, color="black", linewidth=1)
ax3.set_xticks(x)
ax3.set_xticklabels(periods, rotation=15, ha="right")
ax3.set_ylabel("Avg Spread (EUR/MWh)")
ax3.set_title("Average Spread by Period")

for bar, wr in zip(bars, win_rates):
    if wr > 0:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{wr:.0f}%", ha="center", fontsize=10)

# Plot 4: Rolling 7-day average spread
ax4 = axes[1, 1]
daily["rolling_7d"] = daily["spread"].rolling(7, min_periods=1).mean()
ax4.plot(daily["datetime"], daily["rolling_7d"], "b-", linewidth=2, label="7-day rolling avg")
ax4.axhline(y=0, color="black", linewidth=1)
ax4.axhline(y=20, color="green", linestyle="--", alpha=0.5, label="Historical avg (~20)")

for date_str, label in [("2025-12-15", "Dec 15")]:
    date = pd.to_datetime(date_str)
    if date >= daily["datetime"].min() and date <= daily["datetime"].max():
        ax4.axvline(x=date, color="red", linestyle="--", alpha=0.7, label=label)

ax4.set_xlabel("Date")
ax4.set_ylabel("7-Day Rolling Avg Spread (EUR/MWh)")
ax4.set_title("Spread Trend Over Time")
ax4.legend()
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "11_inefficiency_check.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[+] Saved 11_inefficiency_check.png")
