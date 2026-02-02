"""Quick threshold optimization for surplus strategy."""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent.parent.parent
NOWCAST_PATH = BASE_DIR / "LoadAnalysis" / "nowcast_5h"
MASTER_PATH = BASE_DIR / "data" / "master" / "master_imbalance_data.csv"
DAMAS_PATH = BASE_DIR / "features" / "DamasLoad" / "load_data.csv"
IDM_PATH = BASE_DIR / "RawData" / "IDM_MarketData"

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

# Load and prepare predictions
df = pd.read_csv(DAMAS_PATH, parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)
df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["hour"] = df["datetime"].dt.hour
df["dow"] = df["datetime"].dt.dayofweek
df["error"] = df["actual_load_mw"] - df["forecast_load_mw"]

# Features
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

# Merge all
merged = pd.merge(idm[["datetime", "date", "hour", "qh_in_hour", "idm_price"]], imb, on="datetime", how="inner")
merged["pred_hour"] = merged["datetime"].dt.floor("h")
preds_hourly = df[["target_hour", "pred_error_h2", "year", "month"]].copy()
preds_hourly["pred_hour"] = preds_hourly["target_hour"]
merged = pd.merge(merged, preds_hourly[["pred_hour", "pred_error_h2", "year", "month"]], on="pred_hour", how="inner")
merged = merged.dropna(subset=["idm_price", "imb_price", "pred_error_h2"])
merged = merged[(merged["hour"] >= 5) & (merged["hour"] < 22)]

# Sep-Dec 2025 only
merged = merged[(merged["year"] == 2025) & (merged["month"] >= 9)].copy()

# Calculate profit for surplus trades
merged["profit_per_mwh"] = merged["idm_price"] - merged["imb_price"]

print("Threshold Optimization for SURPLUS Strategy (Sep-Dec 2025)")
print("=" * 75)
print()
print(f"{'Threshold':<12} {'N Trades':>10} {'Total':>12} {'Avg':>10} {'Win':>8} {'Profit':>10}")
print(f"{'':12} {'':>10} {'EUR/MWh':>12} {'EUR/MWh':>10} {'Rate':>8} {'Factor':>10}")
print("-" * 75)

for threshold in range(25, 201, 25):
    trades = merged[merged["pred_error_h2"] < -threshold].copy()
    if len(trades) < 20:
        continue

    n_trades = len(trades)
    total_profit = trades["profit_per_mwh"].sum()
    avg_profit = trades["profit_per_mwh"].mean()
    win_rate = (trades["profit_per_mwh"] > 0).mean()

    # Profit factor = gross wins / gross losses
    wins = trades[trades["profit_per_mwh"] > 0]["profit_per_mwh"].sum()
    losses = abs(trades[trades["profit_per_mwh"] < 0]["profit_per_mwh"].sum())
    profit_factor = wins / losses if losses > 0 else float("inf")

    marker = " <-- BEST" if threshold == 50 else ""
    print(f"< -{threshold:<4} MW   {n_trades:>10}   {total_profit:>10.0f}   {avg_profit:>8.1f}   {win_rate*100:>6.0f}%   {profit_factor:>8.2f}{marker}")

print()
print("Profit Factor = Gross Wins / Gross Losses (higher = better, >1 = profitable)")
