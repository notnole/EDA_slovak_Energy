"""Check if signal-based trading is profitable in Jan 2026."""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
MASTER_PATH = BASE_DIR / "data" / "master" / "master_imbalance_data.csv"
IDM_PATH = BASE_DIR / "RawData" / "IDM_MarketData"
DAMAS_PATH = BASE_DIR / "features" / "DamasLoad" / "load_data.csv"
NOWCAST_PATH = BASE_DIR / "LoadAnalysis" / "nowcast_5h"

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

# Load DAMAS and create features for predictions
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

# Filter periods
jan2026 = merged[(merged["year"] == 2026) & (merged["month"] == 1)].copy()
sepdec = merged[(merged["year"] == 2025) & (merged["month"] >= 9)].copy()

print("\n" + "=" * 85)
print("PROFIT COMPARISON: Signal-Based vs Always Trade")
print("=" * 85)

for period_name, data in [("Sep-Dec 2025", sepdec), ("Jan 2026", jan2026)]:
    print(f"\n--- {period_name} ---")
    print(f"{'Strategy':<30} {'N':>8} {'Total EUR/MWh':>14} {'Avg':>10} {'Win':>8}")
    print("-" * 75)

    # Always sell (profit = IDM - Imbalance)
    n = len(data)
    total = data["spread"].sum()
    avg = data["spread"].mean()
    win = (data["spread"] > 0).mean() * 100
    print(f"{'Always Sell':<30} {n:>8} {total:>12.0f}   {avg:>8.1f}   {win:>6.0f}%")

    # Signal: Surplus - sell when pred < -50
    surplus50 = data[data["pred_error_h2"] < -50]
    if len(surplus50) > 0:
        print(f"{'SELL when pred < -50 MW':<30} {len(surplus50):>8} {surplus50['spread'].sum():>12.0f}   {surplus50['spread'].mean():>8.1f}   {(surplus50['spread']>0).mean()*100:>6.0f}%")

    # Signal: Surplus - sell when pred < -100
    surplus100 = data[data["pred_error_h2"] < -100]
    if len(surplus100) > 0:
        print(f"{'SELL when pred < -100 MW':<30} {len(surplus100):>8} {surplus100['spread'].sum():>12.0f}   {surplus100['spread'].mean():>8.1f}   {(surplus100['spread']>0).mean()*100:>6.0f}%")

    # Signal: Deficit - buy when pred > +50 (profit = Imbalance - IDM = -spread)
    deficit50 = data[data["pred_error_h2"] > 50].copy()
    if len(deficit50) > 0:
        deficit50["profit"] = -deficit50["spread"]
        print(f"{'BUY when pred > +50 MW':<30} {len(deficit50):>8} {deficit50['profit'].sum():>12.0f}   {deficit50['profit'].mean():>8.1f}   {(deficit50['profit']>0).mean()*100:>6.0f}%")

    # Signal: Deficit - buy when pred > +100
    deficit100 = data[data["pred_error_h2"] > 100].copy()
    if len(deficit100) > 0:
        deficit100["profit"] = -deficit100["spread"]
        print(f"{'BUY when pred > +100 MW':<30} {len(deficit100):>8} {deficit100['profit'].sum():>12.0f}   {deficit100['profit'].mean():>8.1f}   {(deficit100['profit']>0).mean()*100:>6.0f}%")

    # Combined: trade both directions based on signal
    combined = data.copy()
    combined["trade"] = "none"
    combined.loc[combined["pred_error_h2"] < -100, "trade"] = "sell"
    combined.loc[combined["pred_error_h2"] > 100, "trade"] = "buy"
    combined["profit"] = 0.0
    combined.loc[combined["trade"] == "sell", "profit"] = combined.loc[combined["trade"] == "sell", "spread"]
    combined.loc[combined["trade"] == "buy", "profit"] = -combined.loc[combined["trade"] == "buy", "spread"]

    trades = combined[combined["trade"] != "none"]
    if len(trades) > 0:
        print(f"{'BOTH directions (|pred|>100)':<30} {len(trades):>8} {trades['profit'].sum():>12.0f}   {trades['profit'].mean():>8.1f}   {(trades['profit']>0).mean()*100:>6.0f}%")
