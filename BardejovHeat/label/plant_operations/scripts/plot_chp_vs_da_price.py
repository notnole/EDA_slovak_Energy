"""Analyze CHP electricity generation decisions vs DA electricity price."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "Bardejov"
PLANT_FILE = DATA_DIR / "plant_timeseries.csv"
DA_FILE = DATA_DIR / "da_prices_hourly.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1]

# Load data
print("[*] Loading data...")
plant = pd.read_csv(PLANT_FILE, parse_dates=["datetime"]).set_index("datetime")
da = pd.read_csv(DA_FILE, parse_dates=["datetime"]).set_index("datetime")

# Resample plant to hourly, convert to MW
hourly = plant.resample("h").mean()
hourly["heat_MW"] = hourly["heat_load_kW"] / 1000
hourly["cool_MW"] = hourly["cooling_kW"] / 1000
hourly["elec_MW"] = hourly["electricity_kW"] / 1000

# Join with DA price
merged = hourly.join(da, how="inner").dropna(subset=["elec_MW", "da_price_eur"])
merged["chp_on"] = merged["elec_MW"] > 0.5  # CHP running if > 500 kW

print(f"[+] {len(merged):,} hours with both plant + DA price data")
print(f"[+] CHP on: {merged['chp_on'].sum():,} hours ({merged['chp_on'].mean()*100:.0f}%)")
print(f"[+] CHP off: {(~merged['chp_on']).sum():,} hours ({(~merged['chp_on']).mean()*100:.0f}%)")

# ============================================================
# Stats: price when CHP on vs off
# ============================================================
on = merged[merged["chp_on"]]
off = merged[~merged["chp_on"]]

print(f"\n--- DA price when CHP is ON ---")
print(f"  Mean: {on['da_price_eur'].mean():.1f} EUR/MWh")
print(f"  Median: {on['da_price_eur'].median():.1f} EUR/MWh")
print(f"  10th percentile: {on['da_price_eur'].quantile(0.1):.1f}")
print(f"  90th percentile: {on['da_price_eur'].quantile(0.9):.1f}")

print(f"\n--- DA price when CHP is OFF ---")
print(f"  Mean: {off['da_price_eur'].mean():.1f} EUR/MWh")
print(f"  Median: {off['da_price_eur'].median():.1f} EUR/MWh")
print(f"  10th percentile: {off['da_price_eur'].quantile(0.1):.1f}")
print(f"  90th percentile: {off['da_price_eur'].quantile(0.9):.1f}")

# Electricity revenue estimate
merged["elec_revenue_eur"] = merged["elec_MW"] * merged["da_price_eur"]  # EUR/h
monthly_rev = merged["elec_revenue_eur"].resample("MS").sum()
print(f"\n--- Estimated electricity revenue ---")
for y in sorted(merged.index.year.unique()):
    yr_rev = merged[merged.index.year == y]["elec_revenue_eur"].sum()
    yr_mwh = merged[merged.index.year == y]["elec_MW"].sum()  # MWh (hourly MW * 1h)
    avg_price = yr_rev / yr_mwh if yr_mwh > 0 else 0
    print(f"  {y}: {yr_rev/1000:.0f} kEUR, {yr_mwh:.0f} MWh, avg captured price: {avg_price:.1f} EUR/MWh")

# ============================================================
# Plot 1: CHP on/off by DA price - histogram
# ============================================================
print("\n[*] Plot 1: Price distribution CHP on vs off...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

bins = np.arange(-50, 300, 5)
ax1.hist(on["da_price_eur"].clip(-50, 295), bins=bins, alpha=0.6, color="goldenrod",
         label=f"CHP ON (n={len(on):,})", density=True)
ax1.hist(off["da_price_eur"].clip(-50, 295), bins=bins, alpha=0.6, color="steelblue",
         label=f"CHP OFF (n={len(off):,})", density=True)
ax1.axvline(on["da_price_eur"].mean(), color="goldenrod", linestyle="--", linewidth=2,
            label=f"ON mean: {on['da_price_eur'].mean():.0f} EUR")
ax1.axvline(off["da_price_eur"].mean(), color="steelblue", linestyle="--", linewidth=2,
            label=f"OFF mean: {off['da_price_eur'].mean():.0f} EUR")
ax1.set_xlabel("DA Price (EUR/MWh)")
ax1.set_ylabel("Density")
ax1.set_title("DA Price Distribution: CHP On vs Off")
ax1.legend()
ax1.grid(True, alpha=0.3)

# CHP on rate by price bucket
price_buckets = pd.cut(merged["da_price_eur"], bins=np.arange(-50, 350, 25))
bucket_stats = merged.groupby(price_buckets, observed=True).agg(
    chp_on_rate=("chp_on", "mean"),
    count=("chp_on", "count"),
    avg_elec=("elec_MW", "mean"),
)
bucket_stats = bucket_stats[bucket_stats["count"] >= 10]

ax2.bar(range(len(bucket_stats)), bucket_stats["chp_on_rate"] * 100,
        color="goldenrod", alpha=0.7)
ax2.set_xticks(range(len(bucket_stats)))
ax2.set_xticklabels([str(x) for x in bucket_stats.index], rotation=45, ha="right", fontsize=7)
ax2.set_xlabel("DA Price Bucket (EUR/MWh)")
ax2.set_ylabel("% Hours CHP Running")
ax2.set_title("CHP On Rate by Price Bucket")
ax2.grid(True, alpha=0.3, axis="y")
ax2.set_ylim(0, 105)

fig.suptitle("Bardejov CHP - Electricity Generation vs DA Price", fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "06_chp_vs_da_price_histogram.png", dpi=150)
print("[+] Saved 06_chp_vs_da_price_histogram.png")

# ============================================================
# Plot 2: Scatter - electricity output vs DA price
# ============================================================
print("[*] Plot 2: Scatter - electricity output vs DA price...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Hourly scatter
ax1.scatter(merged["da_price_eur"], merged["elec_MW"], s=2, alpha=0.1,
            color="goldenrod", rasterized=True)
ax1.set_xlabel("DA Price (EUR/MWh)")
ax1.set_ylabel("Electricity Generation (MW)")
ax1.set_title("Hourly: Electricity vs DA Price")
ax1.set_xlim(-50, 300)
ax1.grid(True, alpha=0.3)

# Average electricity by price bucket (finer bins)
fine_buckets = pd.cut(merged["da_price_eur"], bins=np.arange(-50, 300, 10))
fine_stats = merged.groupby(fine_buckets, observed=True).agg(
    avg_elec=("elec_MW", "mean"),
    count=("elec_MW", "count"),
)
fine_stats = fine_stats[fine_stats["count"] >= 5]
x_mid = [(b.left + b.right) / 2 for b in fine_stats.index]

ax2.bar(x_mid, fine_stats["avg_elec"], width=9, color="goldenrod", alpha=0.7)
ax2.set_xlabel("DA Price (EUR/MWh)")
ax2.set_ylabel("Average Electricity Generation (MW)")
ax2.set_title("Average CHP Output by DA Price Bucket")
ax2.grid(True, alpha=0.3, axis="y")

fig.suptitle("Bardejov CHP - Generation Response to DA Price", fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "07_chp_scatter_da_price.png", dpi=150)
print("[+] Saved 07_chp_scatter_da_price.png")

# ============================================================
# Plot 3: Time series - monthly CHP hours and DA price
# ============================================================
print("[*] Plot 3: Monthly CHP operation vs DA price...")
fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

monthly = merged.resample("MS").agg(
    chp_hours=("chp_on", "sum"),
    total_hours=("chp_on", "count"),
    avg_da_price=("da_price_eur", "mean"),
    elec_MWh=("elec_MW", "sum"),
    revenue_kEUR=("elec_revenue_eur", lambda x: x.sum() / 1000),
)
monthly["chp_pct"] = monthly["chp_hours"] / monthly["total_hours"] * 100
monthly["captured_price"] = monthly["revenue_kEUR"] * 1000 / monthly["elec_MWh"].clip(lower=1)

ax = axes[0]
ax.bar(monthly.index, monthly["chp_pct"], width=25, color="goldenrod", alpha=0.7)
ax.set_ylabel("% Hours CHP Running")
ax.set_title("CHP Utilization Rate")
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3, axis="y")

ax = axes[1]
ax.plot(monthly.index, monthly["avg_da_price"], "o-", color="firebrick", markersize=4)
ax.set_ylabel("DA Price (EUR/MWh)")
ax.set_title("Monthly Average DA Price")
ax.grid(True, alpha=0.3)

ax = axes[2]
bars = ax.bar(monthly.index, monthly["revenue_kEUR"], width=25, color="darkgreen", alpha=0.7)
ax.set_ylabel("Revenue (kEUR)")
ax.set_title("Monthly Electricity Revenue (generation x DA price)")
ax.grid(True, alpha=0.3, axis="y")

for a in axes:
    a.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    a.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

fig.suptitle("Bardejov CHP - Monthly Operation vs DA Electricity Price", fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "08_monthly_chp_vs_da_price.png", dpi=150)
print("[+] Saved 08_monthly_chp_vs_da_price.png")

# ============================================================
# Plot 4: Hourly profile - CHP on rate and DA price by hour
# ============================================================
print("[*] Plot 4: Hourly profiles...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

hourly_profile = merged.groupby(merged.index.hour).agg(
    chp_on_rate=("chp_on", "mean"),
    avg_da_price=("da_price_eur", "mean"),
    avg_elec=("elec_MW", "mean"),
)

ax1.bar(hourly_profile.index, hourly_profile["chp_on_rate"] * 100,
        color="goldenrod", alpha=0.7, label="CHP on rate")
ax1b = ax1.twinx()
ax1b.plot(hourly_profile.index, hourly_profile["avg_da_price"], "ro-",
          markersize=5, linewidth=2, label="DA price")
ax1.set_xlabel("Hour of Day")
ax1.set_ylabel("CHP On Rate (%)")
ax1b.set_ylabel("Avg DA Price (EUR/MWh)")
ax1.set_title("CHP On Rate and DA Price by Hour")
ax1.set_xticks(range(0, 24, 3))
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper left")
ax1b.legend(loc="upper right")

# Weekday vs weekend
for label, mask in [("Weekday", merged.index.dayofweek < 5),
                    ("Weekend", merged.index.dayofweek >= 5)]:
    sub = merged[mask]
    profile = sub.groupby(sub.index.hour)["elec_MW"].mean()
    ax2.plot(profile.index, profile.values, "o-", markersize=4, label=label)
ax2.set_xlabel("Hour of Day")
ax2.set_ylabel("Avg Electricity Generation (MW)")
ax2.set_title("CHP Output Profile: Weekday vs Weekend")
ax2.set_xticks(range(0, 24, 3))
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, None)

fig.suptitle("Bardejov CHP - Daily Patterns", fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "09_chp_hourly_profiles.png", dpi=150)
print("[+] Saved 09_chp_hourly_profiles.png")

# ============================================================
# Plot 5: Week detail - CHP output overlaid with DA price
# ============================================================
print("[*] Plot 5: Week detail with DA price overlay...")
fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

# Pick a week in Oct 2024 when CHP was running
week = merged["2024-10-14":"2024-10-20"]

ax = axes[0]
ax.fill_between(week.index, week["heat_MW"], alpha=0.7, color="firebrick", label="Heat to Network")
ax.fill_between(week.index, week["heat_MW"], week["heat_MW"] + week["cool_MW"].fillna(0),
                alpha=0.5, color="steelblue", label="Cooling")
ax.set_ylabel("Thermal Power (MW)")
ax.set_title("Oct 14-20, 2024 - Thermal Output")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_ylim(0, None)

ax = axes[1]
ax.fill_between(week.index, week["elec_MW"], alpha=0.7, color="goldenrod", label="Electricity")
ax2 = ax.twinx()
ax2.plot(week.index, week["da_price_eur"], color="firebrick", linewidth=1.5,
         alpha=0.8, label="DA Price")
ax.set_ylabel("Electricity (MW)")
ax2.set_ylabel("DA Price (EUR/MWh)")
ax.set_title("Electricity Generation vs DA Price")
ax.legend(loc="upper left")
ax2.legend(loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_ylim(0, None)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b"))

fig.suptitle("Bardejov CHP - Week Detail (Oct 2024)", fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "10_week_detail_chp_vs_price.png", dpi=150)
print("[+] Saved 10_week_detail_chp_vs_price.png")

plt.close("all")
print("[+] All done.")
