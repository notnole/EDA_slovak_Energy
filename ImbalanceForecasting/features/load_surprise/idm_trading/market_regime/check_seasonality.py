"""Check if IDM arbitrage is seasonal by comparing all months."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Merge
merged = pd.merge(idm[["datetime", "hour", "idm_price"]], imb, on="datetime", how="inner")
merged = merged.dropna()
merged["year"] = merged["datetime"].dt.year
merged["month"] = merged["datetime"].dt.month
merged = merged[(merged["hour"] >= 5) & (merged["hour"] < 22)]
merged["spread"] = merged["idm_price"] - merged["imb_price"]
merged = merged.sort_values("datetime")

print(f"[+] Total samples: {len(merged):,}")
print(f"    Date range: {merged['datetime'].min()} to {merged['datetime'].max()}")

# Monthly analysis
print("\n" + "=" * 90)
print("Monthly IDM-Imbalance Spread Analysis (Full 2025 + Jan 2026)")
print("=" * 90)
print(f"{'Month':<12} {'N':>8} {'Avg Spread':>12} {'Median':>10} {'Win Rate':>10} {'Total':>12}")
print("-" * 90)

monthly_stats = []
for year in [2025, 2026]:
    for month in range(1, 13):
        subset = merged[(merged["year"] == year) & (merged["month"] == month)]
        if len(subset) < 50:
            continue

        stats = {
            "year": year,
            "month": month,
            "month_name": f"{pd.Timestamp(year=year, month=month, day=1).strftime('%b')} {year}",
            "n": len(subset),
            "avg_spread": subset["spread"].mean(),
            "median_spread": subset["spread"].median(),
            "win_rate": (subset["spread"] > 0).mean() * 100,
            "total": subset["spread"].sum()
        }
        monthly_stats.append(stats)

        print(f"{stats['month_name']:<12} {stats['n']:>8} {stats['avg_spread']:>10.1f}   {stats['median_spread']:>10.1f} {stats['win_rate']:>8.0f}%   {stats['total']:>10.0f}")

print("-" * 90)

# Compare Jan 2025 vs Jan 2026
jan_2025 = merged[(merged["year"] == 2025) & (merged["month"] == 1)]
jan_2026 = merged[(merged["year"] == 2026) & (merged["month"] == 1)]

print("\n" + "=" * 60)
print("JANUARY COMPARISON: 2025 vs 2026")
print("=" * 60)
if len(jan_2025) > 0 and len(jan_2026) > 0:
    print(f"{'Metric':<20} {'Jan 2025':>15} {'Jan 2026':>15}")
    print("-" * 60)
    print(f"{'N samples':<20} {len(jan_2025):>15} {len(jan_2026):>15}")
    print(f"{'Avg Spread':<20} {jan_2025['spread'].mean():>14.1f}  {jan_2026['spread'].mean():>14.1f}")
    print(f"{'Median Spread':<20} {jan_2025['spread'].median():>14.1f}  {jan_2026['spread'].median():>14.1f}")
    print(f"{'Win Rate':<20} {(jan_2025['spread']>0).mean()*100:>13.0f}%  {(jan_2026['spread']>0).mean()*100:>13.0f}%")
    print(f"{'Total':<20} {jan_2025['spread'].sum():>14.0f}  {jan_2026['spread'].sum():>14.0f}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Monthly average spread
ax1 = axes[0, 0]
df_stats = pd.DataFrame(monthly_stats)
colors = ["green" if s > 5 else "orange" if s > 0 else "red" for s in df_stats["avg_spread"]]
bars = ax1.bar(range(len(df_stats)), df_stats["avg_spread"], color=colors, edgecolor="black", alpha=0.8)
ax1.axhline(y=0, color="black", linewidth=1)
ax1.set_xticks(range(len(df_stats)))
ax1.set_xticklabels(df_stats["month_name"], rotation=45, ha="right")
ax1.set_ylabel("Avg Spread (EUR/MWh)")
ax1.set_title("Monthly Average IDM-Imbalance Spread")

# Add win rate labels
for bar, wr in zip(bars, df_stats["win_rate"]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{wr:.0f}%", ha="center", fontsize=8)

# Plot 2: Monthly win rate
ax2 = axes[0, 1]
colors = ["green" if wr > 55 else "orange" if wr > 50 else "red" for wr in df_stats["win_rate"]]
ax2.bar(range(len(df_stats)), df_stats["win_rate"], color=colors, edgecolor="black", alpha=0.8)
ax2.axhline(y=50, color="black", linewidth=1, linestyle="--", label="50% (breakeven)")
ax2.set_xticks(range(len(df_stats)))
ax2.set_xticklabels(df_stats["month_name"], rotation=45, ha="right")
ax2.set_ylabel("Win Rate (%)")
ax2.set_title("Monthly Win Rate")
ax2.legend()
ax2.set_ylim(30, 80)

# Plot 3: Compare Q1 2025 vs Q4 2025
ax3 = axes[1, 0]
q1_2025 = merged[(merged["year"] == 2025) & (merged["month"].isin([1, 2]))]
q4_2025 = merged[(merged["year"] == 2025) & (merged["month"].isin([9, 10, 11, 12]))]

if len(q1_2025) > 0 and len(q4_2025) > 0:
    data = [q1_2025["spread"], q4_2025["spread"]]
    bp = ax3.boxplot(data, labels=["Jan-Feb 2025", "Sep-Dec 2025"], patch_artist=True)
    bp["boxes"][0].set_facecolor("lightblue")
    bp["boxes"][1].set_facecolor("lightgreen")
    ax3.axhline(y=0, color="black", linewidth=1)
    ax3.set_ylabel("Spread (EUR/MWh)")
    ax3.set_title("Spread Distribution: Early 2025 vs Late 2025")

    # Add means
    ax3.scatter([1, 2], [q1_2025["spread"].mean(), q4_2025["spread"].mean()],
                color="red", marker="D", s=100, zorder=5, label="Mean")
    ax3.legend()

# Plot 4: Cumulative by period
ax4 = axes[1, 1]
jan_feb_2025 = merged[(merged["year"] == 2025) & (merged["month"].isin([1, 2]))].copy()
sep_dec_2025 = merged[(merged["year"] == 2025) & (merged["month"] >= 9)].copy()
jan_2026_data = merged[(merged["year"] == 2026) & (merged["month"] == 1)].copy()

if len(jan_feb_2025) > 0:
    jan_feb_2025["cum"] = jan_feb_2025["spread"].cumsum()
    ax4.plot(range(len(jan_feb_2025)), jan_feb_2025["cum"], "b-", label=f"Jan-Feb 2025 (avg: {jan_feb_2025['spread'].mean():.1f})", alpha=0.7)

if len(sep_dec_2025) > 0:
    sep_dec_2025["cum"] = sep_dec_2025["spread"].cumsum()
    ax4.plot(range(len(sep_dec_2025)), sep_dec_2025["cum"], "g-", label=f"Sep-Dec 2025 (avg: {sep_dec_2025['spread'].mean():.1f})", linewidth=2)

if len(jan_2026_data) > 0:
    jan_2026_data["cum"] = jan_2026_data["spread"].cumsum()
    ax4.plot(range(len(jan_2026_data)), jan_2026_data["cum"], "r-", label=f"Jan 2026 (avg: {jan_2026_data['spread'].mean():.1f})", linewidth=2)

ax4.axhline(y=0, color="black", linewidth=0.5)
ax4.set_xlabel("Trade Number")
ax4.set_ylabel("Cumulative Profit (EUR/MWh)")
ax4.set_title("Cumulative P&L by Period")
ax4.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "12_seasonality_check.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[+] Saved 12_seasonality_check.png")
