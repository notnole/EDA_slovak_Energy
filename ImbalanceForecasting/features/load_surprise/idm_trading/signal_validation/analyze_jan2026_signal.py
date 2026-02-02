"""Deep analysis of signal-based trading in Jan 2026."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Load imbalance
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

# Add predictions to merged
merged["pred_hour"] = merged["datetime"].dt.floor("h")
preds_hourly = df[["target_hour", "pred_error_h2"]].copy()
preds_hourly["pred_hour"] = preds_hourly["target_hour"]
merged = pd.merge(merged, preds_hourly[["pred_hour", "pred_error_h2"]], on="pred_hour", how="left")

# Filter to Jan 2026
jan = merged[(merged["year"] == 2026) & (merged["month"] == 1)].copy()
jan["date"] = jan["datetime"].dt.date

# Create trades based on signal
jan["trade_type"] = "none"
jan.loc[jan["pred_error_h2"] < -100, "trade_type"] = "sell"
jan.loc[jan["pred_error_h2"] > 100, "trade_type"] = "buy"

# Calculate profit
jan["profit"] = 0.0
jan.loc[jan["trade_type"] == "sell", "profit"] = jan.loc[jan["trade_type"] == "sell", "spread"]
jan.loc[jan["trade_type"] == "buy", "profit"] = -jan.loc[jan["trade_type"] == "buy", "spread"]

trades = jan[jan["trade_type"] != "none"].copy()

print(f"\n[+] Jan 2026: {len(jan)} total samples, {len(trades)} signal trades")

# ============ ANALYSIS ============
print("\n" + "=" * 80)
print("SIGNAL-BASED TRADING ANALYSIS - JANUARY 2026")
print("=" * 80)

# 1. Overview by trade type
print("\n--- BY TRADE TYPE ---")
print(f"{'Type':<10} {'N':>6} {'Total':>10} {'Avg':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Win%':>6}")
print("-" * 75)

for trade_type in ["sell", "buy"]:
    subset = trades[trades["trade_type"] == trade_type]
    if len(subset) > 0:
        print(f"{trade_type.upper():<10} {len(subset):>6} {subset['profit'].sum():>8.0f}   {subset['profit'].mean():>6.1f}   {subset['profit'].std():>6.1f}   {subset['profit'].min():>6.0f}   {subset['profit'].max():>6.0f}   {(subset['profit']>0).mean()*100:>4.0f}%")

print(f"{'TOTAL':<10} {len(trades):>6} {trades['profit'].sum():>8.0f}   {trades['profit'].mean():>6.1f}   {trades['profit'].std():>6.1f}   {trades['profit'].min():>6.0f}   {trades['profit'].max():>6.0f}   {(trades['profit']>0).mean()*100:>4.0f}%")

# 2. By signal strength
print("\n--- BY SIGNAL STRENGTH ---")
print(f"{'Threshold':<20} {'N':>6} {'Avg':>8} {'Std':>8} {'Win%':>6} {'Sharpe':>8}")
print("-" * 60)

for thresh in [50, 75, 100, 125, 150]:
    subset = jan[(jan["pred_error_h2"] < -thresh) | (jan["pred_error_h2"] > thresh)].copy()
    subset["profit"] = np.where(subset["pred_error_h2"] < -thresh, subset["spread"], -subset["spread"])
    if len(subset) > 5:
        sharpe = subset["profit"].mean() / subset["profit"].std() if subset["profit"].std() > 0 else 0
        print(f"|pred| > {thresh} MW       {len(subset):>6} {subset['profit'].mean():>6.1f}   {subset['profit'].std():>6.1f}   {(subset['profit']>0).mean()*100:>4.0f}%   {sharpe:>6.2f}")

# 3. Daily statistics
print("\n--- DAILY STATISTICS ---")
daily = trades.groupby("date").agg({
    "profit": ["sum", "count", "mean"],
    "trade_type": lambda x: (x == "sell").sum()
}).reset_index()
daily.columns = ["date", "daily_pnl", "n_trades", "avg_profit", "n_sells"]
daily["n_buys"] = daily["n_trades"] - daily["n_sells"]

print(f"Trading days: {len(daily)}")
print(f"Avg trades/day: {daily['n_trades'].mean():.1f}")
print(f"Avg daily P&L: {daily['daily_pnl'].mean():.1f} EUR/MWh")
print(f"Std daily P&L: {daily['daily_pnl'].std():.1f} EUR/MWh")
print(f"Best day: {daily['daily_pnl'].max():.0f} EUR/MWh")
print(f"Worst day: {daily['daily_pnl'].min():.0f} EUR/MWh")
print(f"Profitable days: {(daily['daily_pnl'] > 0).sum()}/{len(daily)} ({(daily['daily_pnl'] > 0).mean()*100:.0f}%)")

# 4. Risk metrics
print("\n--- RISK METRICS ---")
trades_sorted = trades.sort_values("datetime")
trades_sorted["cum_profit"] = trades_sorted["profit"].cumsum()
trades_sorted["running_max"] = trades_sorted["cum_profit"].cummax()
trades_sorted["drawdown"] = trades_sorted["cum_profit"] - trades_sorted["running_max"]

max_drawdown = trades_sorted["drawdown"].min()
total_profit = trades_sorted["profit"].sum()
sharpe = trades["profit"].mean() / trades["profit"].std() if trades["profit"].std() > 0 else 0

# Profit factor
wins = trades[trades["profit"] > 0]["profit"].sum()
losses = abs(trades[trades["profit"] < 0]["profit"].sum())
profit_factor = wins / losses if losses > 0 else float("inf")

# Win/loss ratio
avg_win = trades[trades["profit"] > 0]["profit"].mean() if (trades["profit"] > 0).sum() > 0 else 0
avg_loss = abs(trades[trades["profit"] < 0]["profit"].mean()) if (trades["profit"] < 0).sum() > 0 else 0
win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")

print(f"Total Profit: {total_profit:.0f} EUR/MWh")
print(f"Max Drawdown: {max_drawdown:.0f} EUR/MWh")
print(f"Profit/MaxDD: {abs(total_profit/max_drawdown):.2f}x" if max_drawdown < 0 else "No drawdown")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Profit Factor: {profit_factor:.2f}")
print(f"Avg Win: {avg_win:.1f}, Avg Loss: {avg_loss:.1f}")
print(f"Win/Loss Ratio: {win_loss_ratio:.2f}")

# 5. By hour
print("\n--- BY HOUR ---")
hourly = trades.groupby("hour").agg({
    "profit": ["sum", "count", "mean"],
}).reset_index()
hourly.columns = ["hour", "total", "count", "avg"]
hourly = hourly.sort_values("avg", ascending=False)
print(f"{'Hour':<6} {'N':>6} {'Total':>10} {'Avg':>8}")
print("-" * 35)
for _, row in hourly.head(5).iterrows():
    print(f"{int(row['hour']):>4}h  {int(row['count']):>6} {row['total']:>8.0f}   {row['avg']:>6.1f}")
print("...")
for _, row in hourly.tail(3).iterrows():
    print(f"{int(row['hour']):>4}h  {int(row['count']):>6} {row['total']:>8.0f}   {row['avg']:>6.1f}")

# 6. Opportunity sizing (at 5 MWh position)
print("\n--- OPPORTUNITY @ 5 MWh POSITION ---")
TRADE_SIZE = 5
print(f"Total Profit: {total_profit * TRADE_SIZE:,.0f} EUR")
print(f"Max Drawdown: {max_drawdown * TRADE_SIZE:,.0f} EUR")
print(f"Per Trade Avg: {trades['profit'].mean() * TRADE_SIZE:.0f} EUR")
print(f"Daily Avg: {daily['daily_pnl'].mean() * TRADE_SIZE:.0f} EUR")

# ============ VISUALIZATION ============
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Cumulative P&L
ax1 = axes[0, 0]
ax1.plot(range(len(trades_sorted)), trades_sorted["cum_profit"], "g-", linewidth=1.5)
ax1.fill_between(range(len(trades_sorted)), trades_sorted["cum_profit"], 0,
                  where=trades_sorted["cum_profit"] >= 0, alpha=0.3, color="green")
ax1.fill_between(range(len(trades_sorted)), trades_sorted["cum_profit"], 0,
                  where=trades_sorted["cum_profit"] < 0, alpha=0.3, color="red")
ax1.axhline(y=0, color="black", linewidth=0.5)
ax1.set_xlabel("Trade Number")
ax1.set_ylabel("Cumulative Profit (EUR/MWh)")
ax1.set_title(f"Cumulative P&L (n={len(trades)} trades)")

# Plot 2: Profit distribution
ax2 = axes[0, 1]
ax2.hist(trades["profit"], bins=30, color="steelblue", edgecolor="black", alpha=0.7)
ax2.axvline(x=0, color="black", linewidth=1)
ax2.axvline(x=trades["profit"].mean(), color="green", linewidth=2, linestyle="--",
            label=f"Mean: {trades['profit'].mean():.1f}")
ax2.set_xlabel("Profit per Trade (EUR/MWh)")
ax2.set_ylabel("Count")
ax2.set_title("Profit Distribution")
ax2.legend()

# Plot 3: Daily P&L
ax3 = axes[0, 2]
colors = ["green" if p > 0 else "red" for p in daily["daily_pnl"]]
ax3.bar(range(len(daily)), daily["daily_pnl"], color=colors, alpha=0.7)
ax3.axhline(y=0, color="black", linewidth=1)
ax3.axhline(y=daily["daily_pnl"].mean(), color="blue", linestyle="--",
            label=f"Avg: {daily['daily_pnl'].mean():.1f}")
ax3.set_xlabel("Trading Day")
ax3.set_ylabel("Daily P&L (EUR/MWh)")
ax3.set_title(f"Daily P&L ({(daily['daily_pnl']>0).mean()*100:.0f}% profitable days)")
ax3.legend()

# Plot 4: By signal strength
ax4 = axes[1, 0]
thresholds = [50, 75, 100, 125, 150, 175, 200]
avgs = []
wins = []
counts = []
for thresh in thresholds:
    subset = jan[(jan["pred_error_h2"] < -thresh) | (jan["pred_error_h2"] > thresh)].copy()
    if len(subset) > 3:
        subset["profit"] = np.where(subset["pred_error_h2"] < -thresh, subset["spread"], -subset["spread"])
        avgs.append(subset["profit"].mean())
        wins.append((subset["profit"] > 0).mean() * 100)
        counts.append(len(subset))
    else:
        avgs.append(0)
        wins.append(0)
        counts.append(0)

ax4.bar(range(len(thresholds)), avgs, color="steelblue", alpha=0.8)
ax4.axhline(y=0, color="black", linewidth=1)
ax4.set_xticks(range(len(thresholds)))
ax4.set_xticklabels([f">{t}" for t in thresholds])
ax4.set_xlabel("|Predicted Surprise| Threshold (MW)")
ax4.set_ylabel("Avg Profit (EUR/MWh)")
ax4.set_title("Profit by Signal Strength")

# Add count labels
for i, (avg, cnt) in enumerate(zip(avgs, counts)):
    ax4.text(i, avg + 1, f"n={cnt}", ha="center", fontsize=8)

# Plot 5: Drawdown
ax5 = axes[1, 1]
ax5.fill_between(range(len(trades_sorted)), 0, trades_sorted["drawdown"], color="red", alpha=0.5)
ax5.axhline(y=max_drawdown, color="darkred", linestyle="--", label=f"Max DD: {max_drawdown:.0f}")
ax5.set_xlabel("Trade Number")
ax5.set_ylabel("Drawdown (EUR/MWh)")
ax5.set_title("Drawdown from Peak")
ax5.legend()

# Plot 6: Summary stats
ax6 = axes[1, 2]
ax6.axis("off")

summary = f"""
JANUARY 2026 SIGNAL-BASED TRADING
==================================

OPPORTUNITY (@ 5 MWh position):
  Total Profit:    {total_profit * TRADE_SIZE:>8,.0f} EUR
  Max Drawdown:    {max_drawdown * TRADE_SIZE:>8,.0f} EUR
  Daily Avg:       {daily['daily_pnl'].mean() * TRADE_SIZE:>8,.0f} EUR

RISK METRICS:
  Win Rate:        {(trades['profit']>0).mean()*100:>8.0f}%
  Profit Factor:   {profit_factor:>8.2f}
  Sharpe Ratio:    {sharpe:>8.2f}
  Win/Loss Ratio:  {win_loss_ratio:>8.2f}

TRADE STATISTICS:
  Total Trades:    {len(trades):>8}
  Trading Days:    {len(daily):>8}
  Trades/Day:      {daily['n_trades'].mean():>8.1f}
  Profitable Days: {(daily['daily_pnl']>0).sum():>8}/{len(daily)}

SIGNAL BREAKDOWN:
  SELL signals:    {(trades['trade_type']=='sell').sum():>8}
  BUY signals:     {(trades['trade_type']=='buy').sum():>8}
"""

ax6.text(0.1, 0.95, summary, transform=ax6.transAxes, fontsize=11,
         verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "14_jan2026_signal_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[+] Saved 14_jan2026_signal_analysis.png")
