"""Check when signal-based trading started outperforming always-sell."""
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

imb = pd.read_csv(MASTER_PATH, parse_dates=["datetime"])
imb = imb[["datetime", "System Imbalance (MWh)", "Imbalance Settlement Price (EUR/MWh)"]].copy()
imb.columns = ["datetime", "imbalance", "imb_price"]

merged = pd.merge(idm[["datetime", "hour", "idm_price"]], imb, on="datetime", how="inner")
merged = merged.dropna()
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

merged["pred_hour"] = merged["datetime"].dt.floor("h")
preds_hourly = df[["target_hour", "pred_error_h2"]].copy()
preds_hourly["pred_hour"] = preds_hourly["target_hour"]
merged = pd.merge(merged, preds_hourly[["pred_hour", "pred_error_h2"]], on="pred_hour", how="left")

# Weekly breakdown from Dec 1 to Jan 25
print("\n" + "=" * 100)
print("WEEKLY COMPARISON: Always Sell vs Signal-Based (Dec 2025 - Jan 2026)")
print("=" * 100)

merged["week_start"] = merged["datetime"].dt.to_period("W").dt.start_time

print(f"\n{'Week Starting':<15} {'N':>6} {'Always Sell':>14} {'Signal':>14} {'Signal Wins?':>14} {'Always Win%':>12} {'Signal Win%':>12}")
print("-" * 95)

weeks = merged[(merged["datetime"] >= "2025-12-01") & (merged["datetime"] <= "2026-01-25")].copy()
weeks = weeks.dropna(subset=["pred_error_h2"])

for week_start in sorted(weeks["week_start"].unique()):
    week_data = weeks[weeks["week_start"] == week_start]
    if len(week_data) < 20:
        continue

    # Always sell
    always_total = week_data["spread"].sum()
    always_win = (week_data["spread"] > 0).mean() * 100

    # Signal-based at ±100 MW
    signal = week_data[(week_data["pred_error_h2"] < -100) | (week_data["pred_error_h2"] > 100)].copy()
    if len(signal) < 3:
        signal_total = 0
        signal_win = 0
        n_signal = 0
    else:
        signal["profit"] = np.where(signal["pred_error_h2"] < -100, signal["spread"], -signal["spread"])
        signal_total = signal["profit"].sum()
        signal_win = (signal["profit"] > 0).mean() * 100
        n_signal = len(signal)

    winner = "SIGNAL" if signal_total > always_total else "ALWAYS"

    print(f"{week_start.strftime('%Y-%m-%d'):<15} {len(week_data):>6} {always_total:>12.0f}   {signal_total:>12.0f}   {winner:>14} {always_win:>10.0f}%   {signal_win:>10.0f}%")

# Summary by period
print("\n" + "=" * 100)
print("PERIOD SUMMARY")
print("=" * 100)

periods = [
    ("Dec 1-14", "2025-12-01", "2025-12-15"),
    ("Dec 15-31", "2025-12-15", "2026-01-01"),
    ("Jan 1-15", "2026-01-01", "2026-01-16"),
    ("Jan 16-25", "2026-01-16", "2026-01-26"),
]

print(f"\n{'Period':<15} {'N':>6} {'Always Sell':>14} {'Signal':>14} {'Winner':>10} {'Delta':>12}")
print("-" * 75)

for name, start, end in periods:
    period = merged[(merged["datetime"] >= start) & (merged["datetime"] < end)].copy()
    period = period.dropna(subset=["pred_error_h2"])
    if len(period) < 20:
        continue

    always_total = period["spread"].sum()

    signal = period[(period["pred_error_h2"] < -100) | (period["pred_error_h2"] > 100)].copy()
    if len(signal) >= 3:
        signal["profit"] = np.where(signal["pred_error_h2"] < -100, signal["spread"], -signal["spread"])
        signal_total = signal["profit"].sum()
    else:
        signal_total = 0

    winner = "SIGNAL" if signal_total > always_total else "ALWAYS"
    delta = signal_total - always_total

    print(f"{name:<15} {len(period):>6} {always_total:>12.0f}   {signal_total:>12.0f}   {winner:>10} {delta:>+10.0f}")

print("\n" + "=" * 100)
print("CONCLUSION")
print("=" * 100)
