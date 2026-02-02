"""Plot strategy comparison v5: Smoothed lines + scatter for signal logic."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
MASTER_PATH = BASE_DIR / "data" / "master" / "master_imbalance_data.csv"
IDM_PATH = BASE_DIR / "RawData" / "IDM_MarketData"
DAMAS_PATH = BASE_DIR / "features" / "DamasLoad" / "load_data.csv"
NOWCAST_PATH = BASE_DIR / "LoadAnalysis" / "nowcast_5h"
OUTPUT_DIR = Path(__file__).parent

print("[*] Loading data...")

# Load IDM
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

imb = pd.read_csv(MASTER_PATH, parse_dates=["datetime"])
imb = imb[["datetime", "System Imbalance (MWh)", "Imbalance Settlement Price (EUR/MWh)"]].copy()
imb.columns = ["datetime", "imbalance", "imb_price"]

merged = pd.merge(idm[["datetime", "hour", "idm_price"]], imb, on="datetime", how="inner")
merged = merged.dropna()
merged = merged[(merged["hour"] >= 5) & (merged["hour"] < 22)]
merged["spread"] = merged["idm_price"] - merged["imb_price"]
merged["date"] = merged["datetime"].dt.date

# Load predictions
df = pd.read_csv(DAMAS_PATH, parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)
df["hour"] = df["datetime"].dt.hour
df["dow"] = df["datetime"].dt.dayofweek
df["error"] = df["actual_load_mw"] - df["forecast_load_mw"]

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

model = joblib.load(NOWCAST_PATH / "models" / "stage1_h2.joblib")
valid_mask = df[features].notna().all(axis=1)
preds = np.full(len(df), np.nan)
preds[valid_mask] = model.predict(df.loc[valid_mask, features])
df["pred_error_h2"] = preds
df["target_hour"] = df["datetime"] + pd.Timedelta(hours=2)

merged["pred_hour"] = merged["datetime"].dt.floor("h")
preds_hourly = df[["target_hour", "pred_error_h2"]].copy()
preds_hourly["pred_hour"] = preds_hourly["target_hour"]
merged = pd.merge(merged, preds_hourly[["pred_hour", "pred_error_h2"]], on="pred_hour", how="left")

# Filter Dec-Jan
data = merged[(merged["datetime"] >= "2025-12-01") & (merged["datetime"] <= "2026-01-25")].copy()
data = data.dropna(subset=["pred_error_h2"])

# Calculate signal
data["has_signal"] = (data["pred_error_h2"].abs() > 100)
data["signal_profit"] = np.where(
    data["pred_error_h2"] < -100, data["spread"],
    np.where(data["pred_error_h2"] > 100, -data["spread"], 0)
)

# Create figure
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# ============ PLOT 1: Daily Profits with step lines ============
ax1 = axes[0]

daily_always = data.groupby("date")["spread"].sum().reset_index()
daily_always.columns = ["date", "always_sell"]

daily_signal = data[data["has_signal"]].groupby("date")["signal_profit"].sum().reset_index()
daily_signal.columns = ["date", "signal"]

daily = pd.merge(daily_always, daily_signal, on="date", how="left").fillna(0)
daily["date"] = pd.to_datetime(daily["date"])

# Step line with filled area
ax1.step(daily["date"], daily["always_sell"], where="mid", label="Always Sell", color="blue", linewidth=2)
ax1.step(daily["date"], daily["signal"], where="mid", label="Signal |pred|>100", color="green", linewidth=2)

ax1.fill_between(daily["date"], 0, daily["always_sell"], step="mid",
                  where=daily["always_sell"] >= 0, alpha=0.3, color="blue")
ax1.fill_between(daily["date"], 0, daily["always_sell"], step="mid",
                  where=daily["always_sell"] < 0, alpha=0.3, color="lightcoral")
ax1.fill_between(daily["date"], 0, daily["signal"], step="mid", alpha=0.3, color="green")

ax1.axhline(y=0, color="black", linewidth=1)
ax1.axvline(x=pd.Timestamp("2026-01-01"), color="red", linestyle="--", linewidth=2, label="Jan 1, 2026")

ax1.set_xlabel("Date")
ax1.set_ylabel("Daily Profit (EUR/MWh)")
ax1.set_title("Daily Profit: Always Sell vs Signal-Based Trading")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

# ============ PLOT 2: Step graph with highlighted spread ============
ax2 = axes[1]

daily_prices = data.groupby("date").agg({
    "idm_price": "mean",
    "imb_price": "mean"
}).reset_index()
daily_prices["date"] = pd.to_datetime(daily_prices["date"])

# Step graph
ax2.step(daily_prices["date"], daily_prices["idm_price"], where="mid",
         label="IDM Price", color="#1f77b4", linewidth=2)
ax2.step(daily_prices["date"], daily_prices["imb_price"], where="mid",
         label="Imbalance Price", color="#ff7f0e", linewidth=2)

# Fill between with green/red
ax2.fill_between(daily_prices["date"], daily_prices["idm_price"], daily_prices["imb_price"],
                  where=daily_prices["idm_price"] >= daily_prices["imb_price"],
                  alpha=0.4, color="green", step="mid", label="IDM > Imb (Sell profits)")
ax2.fill_between(daily_prices["date"], daily_prices["idm_price"], daily_prices["imb_price"],
                  where=daily_prices["idm_price"] < daily_prices["imb_price"],
                  alpha=0.4, color="red", step="mid", label="Imb > IDM (Sell loses)")

ax2.axvline(x=pd.Timestamp("2026-01-01"), color="red", linestyle="--", linewidth=2, label="Jan 1, 2026")

ax2.set_xlabel("Date")
ax2.set_ylabel("Daily Avg Price (EUR/MWh)")
ax2.set_title("Daily Average Prices: IDM vs Imbalance")
ax2.legend(loc="upper left", fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

# ============ PLOT 3: Scatter - Prediction vs Spread with signal zones ============
ax3 = axes[2]

# All data points
ax3.scatter(data["spread"], data["pred_error_h2"], alpha=0.15, s=10, c="gray", label="No signal")

# Highlight signal zones
sell_zone = data[data["pred_error_h2"] < -100]
buy_zone = data[data["pred_error_h2"] > 100]

# Color by profit: SELL profits when spread > 0, BUY profits when spread < 0
sell_profit = sell_zone[sell_zone["spread"] > 0]
sell_loss = sell_zone[sell_zone["spread"] <= 0]
buy_profit = buy_zone[buy_zone["spread"] < 0]
buy_loss = buy_zone[buy_zone["spread"] >= 0]

ax3.scatter(sell_profit["spread"], sell_profit["pred_error_h2"], alpha=0.7, s=25,
            c="green", marker="v", label=f"SELL + Profit (n={len(sell_profit)})")
ax3.scatter(sell_loss["spread"], sell_loss["pred_error_h2"], alpha=0.7, s=25,
            c="lightgreen", marker="v", label=f"SELL + Loss (n={len(sell_loss)})")
ax3.scatter(buy_profit["spread"], buy_profit["pred_error_h2"], alpha=0.7, s=25,
            c="red", marker="^", label=f"BUY + Profit (n={len(buy_profit)})")
ax3.scatter(buy_loss["spread"], buy_loss["pred_error_h2"], alpha=0.7, s=25,
            c="lightcoral", marker="^", label=f"BUY + Loss (n={len(buy_loss)})")

# Add horizontal lines for signal thresholds
ax3.axhline(y=-100, color="green", linestyle="--", linewidth=1.5, alpha=0.7)
ax3.axhline(y=100, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
ax3.axvline(x=0, color="black", linewidth=1)

# Shade profitable zones
ax3.axhspan(-300, -100, xmin=0.5, xmax=1.0, alpha=0.1, color="green")  # SELL zone, spread > 0
ax3.axhspan(100, 300, xmin=0.0, xmax=0.5, alpha=0.1, color="red")  # BUY zone, spread < 0

# Add annotations
ax3.annotate("SELL zone\n(pred < -100)", xy=(-150, -150), fontsize=10, color="darkgreen", fontweight="bold")
ax3.annotate("BUY zone\n(pred > +100)", xy=(-150, 150), fontsize=10, color="darkred", fontweight="bold")
ax3.annotate("Profitable\nSELLs", xy=(80, -200), fontsize=9, color="green", ha="center")
ax3.annotate("Profitable\nBUYs", xy=(-80, 200), fontsize=9, color="red", ha="center")

ax3.set_xlabel("Spread: IDM - Imbalance (EUR/MWh)")
ax3.set_ylabel("Predicted Load Error (MW)")
ax3.set_title("Signal Strategy: Prediction vs Spread (colored by profit/loss)")
ax3.legend(loc="upper right", fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-200, 200)
ax3.set_ylim(-250, 250)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "16_strategy_comparison_v5.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\n[+] Saved 16_strategy_comparison_v5.png")

# Print stats
print("\nSignal Performance:")
print(f"  SELL signals: {len(sell_zone)} total, {len(sell_profit)} profitable ({100*len(sell_profit)/len(sell_zone):.0f}%)")
print(f"  BUY signals:  {len(buy_zone)} total, {len(buy_profit)} profitable ({100*len(buy_profit)/len(buy_zone):.0f}%)")
