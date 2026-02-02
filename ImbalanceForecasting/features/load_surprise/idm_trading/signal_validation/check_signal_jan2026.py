"""Check if load surprise prediction still indicates imbalance direction in Jan 2026."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
NOWCAST_PATH = BASE_DIR / "LoadAnalysis" / "nowcast_5h"
MASTER_PATH = BASE_DIR / "data" / "master" / "master_imbalance_data.csv"
DAMAS_PATH = BASE_DIR / "features" / "DamasLoad" / "load_data.csv"
IDM_PATH = BASE_DIR / "RawData" / "IDM_MarketData"
OUTPUT_DIR = Path(__file__).parent

print("[*] Loading data...")

# Load DAMAS and create features
df = pd.read_csv(DAMAS_PATH, parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)
df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["hour"] = df["datetime"].dt.hour
df["dow"] = df["datetime"].dt.dayofweek
df["error"] = df["actual_load_mw"] - df["forecast_load_mw"]

print(f"  DAMAS: {df['datetime'].min()} to {df['datetime'].max()}")

# Load 3-min features
load_3min = pd.read_csv(BASE_DIR / "data" / "features" / "load_3min.csv")
load_3min["datetime"] = pd.to_datetime(load_3min["datetime"])
load_3min["hour_start"] = load_3min["datetime"].dt.floor("h")
load_hourly = load_3min.groupby("hour_start").agg({"load_mw": ["std", "first", "last"]}).reset_index()
load_hourly.columns = ["datetime", "load_std_3min", "load_first", "load_last"]
load_hourly["load_trend_3min"] = load_hourly["load_last"] - load_hourly["load_first"]
df = df.merge(load_hourly[["datetime", "load_std_3min", "load_trend_3min"]], on="datetime", how="left")

reg_3min = pd.read_csv(BASE_DIR / "data" / "features" / "regulation_3min.csv")
reg_3min["datetime"] = pd.to_datetime(reg_3min["datetime"])
reg_3min["hour_start"] = reg_3min["datetime"].dt.floor("h")
reg_hourly = reg_3min.groupby("hour_start").agg({"regulation_mw": ["mean", "std"]}).reset_index()
reg_hourly.columns = ["datetime", "reg_mean", "reg_std"]
df = df.merge(reg_hourly, on="datetime", how="left")

# Create features
for lag in range(1, 9):
    df[f"error_lag{lag}"] = df["error"].shift(lag)
for window in [3, 6, 12, 24]:
    df[f"error_roll_mean_{window}h"] = df["error"].shift(1).rolling(window).mean()
    df[f"error_roll_std_{window}h"] = df["error"].shift(1).rolling(window).std()
df["error_trend_3h"] = df["error_lag1"] - df["error_lag3"]
df["error_trend_6h"] = df["error_lag1"] - df["error_lag6"]
df["error_momentum"] = (0.5 * (df["error_lag1"] - df["error_lag2"]) +
                        0.3 * (df["error_lag2"] - df["error_lag3"]) +
                        0.2 * (df["error_lag3"] - df["error_lag4"]))
df["load_volatility_lag1"] = df["load_std_3min"].shift(1)
df["load_trend_lag1"] = df["load_trend_3min"].shift(1)
for lag in range(1, 4):
    df[f"reg_mean_lag{lag}"] = df["reg_mean"].shift(lag)
df["reg_std_lag1"] = df["reg_std"].shift(1)
seasonal = df.groupby(["dow", "hour"])["error"].mean().reset_index()
seasonal.columns = ["dow", "hour", "seasonal_error"]
df = df.merge(seasonal, on=["dow", "hour"], how="left")
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["is_weekend"] = (df["dow"] >= 5).astype(int)

features = ["error_lag1", "error_lag2", "error_lag3", "error_lag4", "error_lag5", "error_lag6",
    "error_roll_mean_3h", "error_roll_std_3h", "error_roll_mean_6h", "error_roll_std_6h",
    "error_roll_mean_12h", "error_roll_std_12h", "error_roll_mean_24h", "error_roll_std_24h",
    "error_trend_3h", "error_trend_6h", "error_momentum", "load_volatility_lag1", "load_trend_lag1",
    "reg_mean_lag1", "reg_mean_lag2", "reg_mean_lag3", "reg_std_lag1", "seasonal_error",
    "hour", "hour_sin", "hour_cos", "dow", "is_weekend"]

# Generate H+2 predictions
model = joblib.load(NOWCAST_PATH / "models" / "stage1_h2.joblib")
valid_mask = df[features].notna().all(axis=1)
preds = np.full(len(df), np.nan)
preds[valid_mask] = model.predict(df.loc[valid_mask, features])
df["pred_error_h2"] = preds
df["target_hour"] = df["datetime"] + pd.Timedelta(hours=2)

# Load imbalance
imb = pd.read_csv(MASTER_PATH, parse_dates=["datetime"])
imb = imb[["datetime", "System Imbalance (MWh)", "Imbalance Settlement Price (EUR/MWh)"]].copy()
imb.columns = ["datetime", "imbalance", "imb_price"]
imb = imb[imb["imbalance"].abs() <= 300]
imb["hour_start"] = imb["datetime"].dt.floor("h")
imb_hourly = imb.groupby("hour_start").agg({"imbalance": "mean", "imb_price": "mean"}).reset_index()
imb_hourly.columns = ["target_hour", "future_imbalance", "future_price"]

print(f"  Imbalance: {imb['datetime'].min()} to {imb['datetime'].max()}")

# Merge predictions with future imbalance
merged = pd.merge(
    df[["datetime", "pred_error_h2", "target_hour", "hour", "year", "month"]],
    imb_hourly,
    on="target_hour", how="inner"
)
merged = merged.dropna(subset=["pred_error_h2", "future_imbalance"])
merged = merged[(merged["hour"] >= 5) & (merged["hour"] < 22)]

# Define periods
def get_period(row):
    if row["year"] == 2025:
        if row["month"] < 9:
            return "Jan-Aug 2025"
        else:
            return "Sep-Dec 2025"
    else:
        return "Jan 2026"

merged["period"] = merged.apply(get_period, axis=1)

print(f"\n[+] Total merged samples: {len(merged):,}")

# Analyze direction prediction by period
print("\n" + "=" * 90)
print("DOES LOAD SURPRISE PREDICTION INDICATE IMBALANCE DIRECTION?")
print("=" * 90)

for period in ["Jan-Aug 2025", "Sep-Dec 2025", "Jan 2026"]:
    subset = merged[merged["period"] == period]
    if len(subset) < 100:
        continue

    print(f"\n--- {period} (n={len(subset):,}) ---")
    print(f"{'Threshold':<15} {'N':>8} {'Avg Imb':>12} {'% Positive':>12} {'% Negative':>12}")
    print("-" * 60)

    # Check if negative prediction -> positive imbalance (surplus)
    for thresh in [50, 100, 150]:
        surplus_pred = subset[subset["pred_error_h2"] < -thresh]
        deficit_pred = subset[subset["pred_error_h2"] > thresh]

        if len(surplus_pred) >= 10:
            pct_pos = (surplus_pred["future_imbalance"] > 0).mean() * 100
            avg_imb = surplus_pred["future_imbalance"].mean()
            print(f"Pred < -{thresh} MW  {len(surplus_pred):>8} {avg_imb:>10.1f}   {pct_pos:>10.0f}%   {100-pct_pos:>10.0f}%")

        if len(deficit_pred) >= 10:
            pct_neg = (deficit_pred["future_imbalance"] < 0).mean() * 100
            avg_imb = deficit_pred["future_imbalance"].mean()
            print(f"Pred > +{thresh} MW  {len(deficit_pred):>8} {avg_imb:>10.1f}   {100-pct_neg:>10.0f}%   {pct_neg:>10.0f}%")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Imbalance by prediction bin - comparison across periods
ax1 = axes[0, 0]
bins = [-300, -100, -50, 0, 50, 100, 300]
periods_to_plot = ["Sep-Dec 2025", "Jan 2026"]
colors = ["green", "red"]
width = 0.35
x_labels = []

for i, period in enumerate(periods_to_plot):
    subset = merged[merged["period"] == period]
    subset["bin"] = pd.cut(subset["pred_error_h2"], bins=bins)
    bin_stats = subset.groupby("bin", observed=True)["future_imbalance"].mean()

    x = np.arange(len(bin_stats))
    if i == 0:
        x_labels = [f"{int(b.left)} to {int(b.right)}" for b in bin_stats.index]

    ax1.bar(x + i*width - width/2, bin_stats.values, width, label=period, color=colors[i], alpha=0.7)

ax1.axhline(y=0, color="black", linewidth=1)
ax1.set_xticks(np.arange(len(x_labels)))
ax1.set_xticklabels(x_labels, rotation=45, ha="right")
ax1.set_xlabel("Predicted Load Surprise (MW)")
ax1.set_ylabel("Avg Future Imbalance (MWh)")
ax1.set_title("Does Prediction Indicate Direction?")
ax1.legend()

# Plot 2: Correlation by period
ax2 = axes[0, 1]
correlations = []
for period in ["Jan-Aug 2025", "Sep-Dec 2025", "Jan 2026"]:
    subset = merged[merged["period"] == period]
    if len(subset) > 100:
        corr = subset["pred_error_h2"].corr(subset["future_imbalance"])
        correlations.append({"period": period, "corr": corr, "n": len(subset)})

corr_df = pd.DataFrame(correlations)
colors = ["green" if c < -0.1 else "orange" if c < 0 else "red" for c in corr_df["corr"]]
bars = ax2.bar(corr_df["period"], corr_df["corr"], color=colors, edgecolor="black", alpha=0.8)
ax2.axhline(y=0, color="black", linewidth=1)
ax2.set_ylabel("Correlation")
ax2.set_title("Prediction vs Imbalance Correlation by Period")

for bar, n in zip(bars, corr_df["n"]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.02,
            f"n={n}", ha="center", fontsize=9)

# Plot 3: Direction accuracy at -100 MW threshold
ax3 = axes[1, 0]
accuracy_data = []
for period in ["Jan-Aug 2025", "Sep-Dec 2025", "Jan 2026"]:
    subset = merged[merged["period"] == period]
    surplus = subset[subset["pred_error_h2"] < -100]
    if len(surplus) >= 10:
        acc = (surplus["future_imbalance"] > 0).mean() * 100
        accuracy_data.append({"period": period, "accuracy": acc, "n": len(surplus), "type": "Surplus (< -100 MW)"})

    deficit = subset[subset["pred_error_h2"] > 100]
    if len(deficit) >= 10:
        acc = (deficit["future_imbalance"] < 0).mean() * 100
        accuracy_data.append({"period": period, "accuracy": acc, "n": len(deficit), "type": "Deficit (> +100 MW)"})

acc_df = pd.DataFrame(accuracy_data)
x = np.arange(len(acc_df["period"].unique()))
width = 0.35

for i, trade_type in enumerate(["Surplus (< -100 MW)", "Deficit (> +100 MW)"]):
    data = acc_df[acc_df["type"] == trade_type]
    color = "green" if "Surplus" in trade_type else "red"
    ax3.bar(x + i*width - width/2, data["accuracy"], width, label=trade_type, color=color, alpha=0.7)

ax3.axhline(y=50, color="black", linewidth=1, linestyle="--", label="50% (random)")
ax3.set_xticks(x)
ax3.set_xticklabels(acc_df["period"].unique())
ax3.set_ylabel("Direction Accuracy (%)")
ax3.set_title("Direction Prediction Accuracy at ±100 MW Threshold")
ax3.legend()
ax3.set_ylim(40, 90)

# Plot 4: Summary text
ax4 = axes[1, 1]
ax4.axis("off")

# Calculate summary stats
jan2026 = merged[merged["period"] == "Jan 2026"]
sepdec = merged[merged["period"] == "Sep-Dec 2025"]

summary = "SUMMARY: Load Prediction as Direction Signal\n"
summary += "=" * 45 + "\n\n"

if len(sepdec) > 100:
    corr_sepdec = sepdec["pred_error_h2"].corr(sepdec["future_imbalance"])
    surplus_acc_sepdec = (sepdec[sepdec["pred_error_h2"] < -100]["future_imbalance"] > 0).mean() * 100
    summary += f"Sep-Dec 2025:\n"
    summary += f"  Correlation: {corr_sepdec:.3f}\n"
    summary += f"  Surplus direction acc: {surplus_acc_sepdec:.0f}%\n\n"

if len(jan2026) > 100:
    corr_jan = jan2026["pred_error_h2"].corr(jan2026["future_imbalance"])
    surplus_pred = jan2026[jan2026["pred_error_h2"] < -100]
    if len(surplus_pred) >= 10:
        surplus_acc_jan = (surplus_pred["future_imbalance"] > 0).mean() * 100
    else:
        surplus_acc_jan = float("nan")
    summary += f"Jan 2026:\n"
    summary += f"  Correlation: {corr_jan:.3f}\n"
    summary += f"  Surplus direction acc: {surplus_acc_jan:.0f}%\n\n"

    if corr_jan < -0.1:
        summary += "CONCLUSION: Prediction STILL indicates direction\n"
        summary += "in Jan 2026 (negative correlation preserved)"
    else:
        summary += "CONCLUSION: Prediction signal WEAKENED\n"
        summary += "in Jan 2026"

ax4.text(0.1, 0.9, summary, transform=ax4.transAxes, fontsize=12,
         verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "13_signal_check_jan2026.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[+] Saved 13_signal_check_jan2026.png")
