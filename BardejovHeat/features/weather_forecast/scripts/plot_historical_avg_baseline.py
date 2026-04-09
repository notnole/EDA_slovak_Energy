"""Evaluate the historical average baseline: predict heat load as the mean of
the same (month, day-of-month, hour) from all prior years."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy import stats

# Paths
DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "Bardejov"
HEAT_FILE = DATA_DIR / "heat_load_timeseries.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1]

# Load heat load -> hourly MW
print("[*] Loading heat load data...")
df = pd.read_csv(HEAT_FILE, parse_dates=["datetime"])
df = df.set_index("datetime").sort_index()
hourly_MW = df["heat_load_kW"].resample("h").mean() / 1000

# Build the historical average baseline:
# For each hour, predict the mean of that (month, day, hour) across all prior years
hourly_MW = hourly_MW.to_frame("actual")
hourly_MW["month"] = hourly_MW.index.month
hourly_MW["day"] = hourly_MW.index.day
hourly_MW["hour"] = hourly_MW.index.hour
hourly_MW["year"] = hourly_MW.index.year
hourly_MW["doy"] = hourly_MW.index.dayofyear

# Leave-one-year-out: for each year, the prediction is the mean of all other years
# at the same (month, day, hour)
print("[*] Computing leave-one-year-out historical average...")
group_cols = ["month", "day", "hour"]
predictions = []

for year in sorted(hourly_MW["year"].unique()):
    train = hourly_MW[hourly_MW["year"] != year]
    test = hourly_MW[hourly_MW["year"] == year].copy()

    # Mean of same (month, day, hour) from other years
    avg = train.groupby(group_cols)["actual"].mean().rename("predicted")
    test = test.join(avg, on=group_cols)
    predictions.append(test)

result = pd.concat(predictions).sort_index()
result = result.dropna(subset=["actual", "predicted"])

print(f"[+] {len(result):,} hourly predictions")

# Overall metrics
mae = np.abs(result["actual"] - result["predicted"]).mean()
rmse = np.sqrt(((result["actual"] - result["predicted"]) ** 2).mean())
r_val = result["actual"].corr(result["predicted"])
bias = (result["predicted"] - result["actual"]).mean()

print(f"[+] Overall: MAE={mae:.2f} MW, RMSE={rmse:.2f} MW, r={r_val:.3f}, Bias={bias:+.2f} MW")

# Daily aggregation
daily = result.groupby(result.index.date).agg(
    actual_MWh=("actual", lambda x: x.mean() * 24),
    predicted_MWh=("predicted", lambda x: x.mean() * 24),
)
daily.index = pd.to_datetime(daily.index)
daily["error_MWh"] = daily["predicted_MWh"] - daily["actual_MWh"]

mae_daily = np.abs(daily["error_MWh"]).mean()
rmse_daily = np.sqrt((daily["error_MWh"] ** 2).mean())
r_daily = daily["actual_MWh"].corr(daily["predicted_MWh"])
mape_daily = (np.abs(daily["error_MWh"]) / daily["actual_MWh"].clip(lower=1)).mean() * 100

print(f"[+] Daily: MAE={mae_daily:.1f} MWh, RMSE={rmse_daily:.1f} MWh, r={r_daily:.3f}, MAPE={mape_daily:.1f}%")

# ============================================================
# Plot 1: Scatter - actual vs predicted (hourly)
# ============================================================
print("[*] Plot 1: Hourly scatter...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Hourly
ax1.scatter(result["actual"], result["predicted"], s=1, alpha=0.1, color="steelblue", rasterized=True)
lims = [0, max(result["actual"].max(), result["predicted"].max()) + 1]
ax1.plot(lims, lims, "k--", linewidth=1, alpha=0.5)
ax1.set_xlabel("Actual Heat Load (MW)")
ax1.set_ylabel("Predicted Heat Load (MW)")
ax1.set_title("Hourly")
ax1.set_xlim(lims)
ax1.set_ylim(lims)
ax1.set_aspect("equal")
ax1.grid(True, alpha=0.3)
stats_text = f"n={len(result):,}\nr={r_val:.3f}\nMAE={mae:.2f} MW\nRMSE={rmse:.2f} MW"
ax1.text(0.97, 0.03, stats_text, transform=ax1.transAxes, ha="right", va="bottom",
         fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
         family="monospace")

# Daily
ax2.scatter(daily["actual_MWh"], daily["predicted_MWh"], s=8, alpha=0.4, color="firebrick")
lims_d = [0, max(daily["actual_MWh"].max(), daily["predicted_MWh"].max()) + 10]
ax2.plot(lims_d, lims_d, "k--", linewidth=1, alpha=0.5)
ax2.set_xlabel("Actual Daily Energy (MWh)")
ax2.set_ylabel("Predicted Daily Energy (MWh)")
ax2.set_title("Daily")
ax2.set_xlim(lims_d)
ax2.set_ylim(lims_d)
ax2.set_aspect("equal")
ax2.grid(True, alpha=0.3)
stats_text_d = f"n={len(daily):,}\nr={r_daily:.3f}\nMAE={mae_daily:.1f} MWh\nMAPE={mape_daily:.1f}%"
ax2.text(0.97, 0.03, stats_text_d, transform=ax2.transAxes, ha="right", va="bottom",
         fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
         family="monospace")

fig.suptitle("Historical Average Baseline - Predicted vs Actual Heat Load", fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "03_baseline_scatter.png", dpi=150)
print("[+] Saved 03_baseline_scatter.png")

# ============================================================
# Plot 2: Time series - actual vs predicted (daily, one year)
# ============================================================
print("[*] Plot 2: Time series overlay...")
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Pick 2024 as example year (most complete)
year_data = daily[daily.index.year == 2024]

ax = axes[0]
ax.plot(year_data.index, year_data["actual_MWh"], linewidth=1, color="firebrick", label="Actual")
ax.plot(year_data.index, year_data["predicted_MWh"], linewidth=1, color="steelblue", alpha=0.8, label="Historical Avg")
ax.set_ylabel("Daily Energy (MWh)")
ax.set_title("2024 - Actual vs Historical Average Baseline")
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, None)

ax = axes[1]
ax.bar(year_data.index, year_data["error_MWh"], width=1, color="gray", alpha=0.6)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_ylabel("Error: Predicted - Actual (MWh)")
ax.set_title("Daily Prediction Error (positive = over-prediction)")
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "04_baseline_timeseries_2024.png", dpi=150)
print("[+] Saved 04_baseline_timeseries_2024.png")

# ============================================================
# Plot 3: Error by month
# ============================================================
print("[*] Plot 3: Error by month...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# MAE by month
monthly_mae = result.copy()
monthly_mae["abs_error"] = np.abs(monthly_mae["actual"] - monthly_mae["predicted"])
mae_by_month = monthly_mae.groupby("month")["abs_error"].mean()

ax1.bar(range(1, 13), mae_by_month.values, color="firebrick", alpha=0.7)
ax1.set_xlabel("Month")
ax1.set_ylabel("MAE (MW)")
ax1.set_title("Hourly MAE by Month")
ax1.set_xticks(range(1, 13))
ax1.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
ax1.grid(True, alpha=0.3, axis="y")

# MAPE by month (daily)
daily_with_month = daily.copy()
daily_with_month["month"] = daily_with_month.index.month
daily_with_month["abs_pct_error"] = (np.abs(daily_with_month["error_MWh"])
                                      / daily_with_month["actual_MWh"].clip(lower=1)) * 100
mape_by_month = daily_with_month.groupby("month")["abs_pct_error"].mean()

ax2.bar(range(1, 13), mape_by_month.values, color="steelblue", alpha=0.7)
ax2.set_xlabel("Month")
ax2.set_ylabel("MAPE (%)")
ax2.set_title("Daily MAPE by Month")
ax2.set_xticks(range(1, 13))
ax2.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
ax2.grid(True, alpha=0.3, axis="y")

fig.suptitle("Historical Average Baseline - Error by Month", fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "05_baseline_error_by_month.png", dpi=150)
print("[+] Saved 05_baseline_error_by_month.png")

plt.close("all")
print("[+] All done.")
