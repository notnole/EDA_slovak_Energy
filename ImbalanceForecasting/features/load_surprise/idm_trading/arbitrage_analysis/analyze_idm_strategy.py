"""
IDM Trading Strategy Analysis
==============================
Compare IDM prices vs Imbalance prices for QH1-2 based on load surprise predictions.

Strategy:
- SURPLUS predicted (surprise < -100 MW): SELL on IDM, settle at imbalance
  -> Profit = IDM Price - Imbalance Price
- DEFICIT predicted (surprise > +100 MW): BUY on IDM, settle at imbalance
  -> Profit = Imbalance Price - IDM Price
"""

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


def load_idm_data():
    """Load all IDM 15-min data."""
    print("[*] Loading IDM data...")

    all_data = []
    for folder in IDM_PATH.iterdir():
        if folder.is_dir() and folder.name.startswith("IDM_total"):
            csv_path = folder / "15 min.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path, sep=";", decimal=",")
                all_data.append(df)
                print(f"  [+] Loaded {folder.name}")

    idm = pd.concat(all_data, ignore_index=True)

    # Parse datetime
    idm["date"] = pd.to_datetime(idm["Delivery day"], format="%d.%m.%Y")
    idm["period_num"] = idm["Period number"]
    idm["hour"] = (idm["period_num"] - 1) // 4
    idm["qh_in_hour"] = ((idm["period_num"] - 1) % 4) + 1
    idm["datetime"] = idm["date"] + pd.to_timedelta(idm["hour"], unit="h") + pd.to_timedelta((idm["qh_in_hour"]-1)*15, unit="m")

    # Get IDM price
    idm["idm_price"] = pd.to_numeric(idm["Weighted average price of all trades (EUR/MWh)"], errors="coerce")
    idm["idm_volume"] = pd.to_numeric(idm["Total Traded Quantity (MW)"], errors="coerce")

    # Filter to QH1-2 only
    idm = idm[idm["qh_in_hour"].isin([1, 2])].copy()

    print(f"[+] Total IDM records (QH1-2): {len(idm):,}")
    print(f"    Date range: {idm['date'].min()} to {idm['date'].max()}")

    return idm[["datetime", "date", "hour", "qh_in_hour", "idm_price", "idm_volume"]]


def load_imbalance_prices():
    """Load imbalance settlement prices."""
    print("[*] Loading imbalance prices...")

    imb = pd.read_csv(MASTER_PATH, parse_dates=["datetime"])
    imb = imb[["datetime", "System Imbalance (MWh)", "Imbalance Settlement Price (EUR/MWh)"]].copy()
    imb.columns = ["datetime", "imbalance", "imb_price"]
    imb = imb[imb["imbalance"].abs() <= 300]

    print(f"[+] Loaded {len(imb):,} imbalance records")
    return imb


def load_predictions():
    """Load H+2 predictions (same as before)."""
    print("[*] Loading predictions...")

    df = pd.read_csv(DAMAS_PATH, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["hour"] = df["datetime"].dt.hour
    df["dow"] = df["datetime"].dt.dayofweek
    df["error"] = df["actual_load_mw"] - df["forecast_load_mw"]

    # Load 3-min data
    load_3min = pd.read_csv(BASE_DIR / "data" / "features" / "load_3min.csv")
    load_3min["datetime"] = pd.to_datetime(load_3min["datetime"])
    load_3min["hour_start"] = load_3min["datetime"].dt.floor("h")
    load_hourly = load_3min.groupby("hour_start").agg({"load_mw": ["std", "first", "last"]}).reset_index()
    load_hourly.columns = ["datetime", "load_std_3min", "load_first", "load_last"]
    load_hourly["load_trend_3min"] = load_hourly["load_last"] - load_hourly["load_first"]
    df = df.merge(load_hourly[["datetime", "load_std_3min", "load_trend_3min"]], on="datetime", how="left")

    # Regulation
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

    # H+2 target hour
    df["target_hour"] = df["datetime"] + pd.Timedelta(hours=2)

    print(f"[+] Generated predictions for {valid_mask.sum():,} hours")
    return df[["datetime", "target_hour", "pred_error_h2", "year", "month", "hour"]]


def analyze_strategy(idm, imb, preds):
    """Analyze trading strategy."""
    print("[*] Analyzing strategy...")

    # Merge IDM with imbalance prices
    merged = pd.merge(idm, imb, on="datetime", how="inner")

    # Merge with predictions (prediction is for the hour, matches to QH1-2)
    merged["pred_hour"] = merged["datetime"].dt.floor("h")
    preds_hourly = preds.copy()
    preds_hourly["pred_hour"] = preds_hourly["target_hour"]

    merged = pd.merge(
        merged,
        preds_hourly[["pred_hour", "pred_error_h2", "year", "month"]],
        on="pred_hour",
        how="inner"
    )

    merged = merged.dropna(subset=["idm_price", "imb_price", "pred_error_h2"])

    # Filter to day hours
    merged = merged[(merged["hour"] >= 5) & (merged["hour"] < 22)]

    # Filter to 2025 only (out-of-sample)
    merged = merged[merged["year"] == 2025].copy()

    print(f"[+] Merged samples: {len(merged):,}")

    # Calculate profits
    # SURPLUS (sell IDM, settle imbalance): profit = IDM - Imbalance
    # DEFICIT (buy IDM, settle imbalance): profit = Imbalance - IDM

    merged["profit_if_surplus_trade"] = merged["idm_price"] - merged["imb_price"]
    merged["profit_if_deficit_trade"] = merged["imb_price"] - merged["idm_price"]

    # Period split
    def get_period(row):
        if row["month"] < 9:
            return "Jan-Aug 2025"
        else:
            return "Sep-Dec 2025"

    merged["period"] = merged.apply(get_period, axis=1)

    return merged


def print_results(merged):
    """Print strategy results."""

    for period in ["Jan-Aug 2025", "Sep-Dec 2025"]:
        subset = merged[merged["period"] == period]
        if len(subset) < 100:
            continue

        print()
        print("=" * 60)
        print(f"{period} (n={len(subset):,} QH samples)")
        print("=" * 60)

        baseline_idm = subset["idm_price"].mean()
        baseline_imb = subset["imb_price"].mean()
        print(f"Baseline IDM price:       {baseline_idm:.1f} EUR/MWh")
        print(f"Baseline Imbalance price: {baseline_imb:.1f} EUR/MWh")
        print()

        for threshold in [50, 100, 150]:
            surplus_trades = subset[subset["pred_error_h2"] < -threshold]
            deficit_trades = subset[subset["pred_error_h2"] > threshold]

            print(f"--- Threshold: +/-{threshold} MW ---")
            print()

            if len(surplus_trades) >= 10:
                avg_profit = surplus_trades["profit_if_surplus_trade"].mean()
                avg_idm = surplus_trades["idm_price"].mean()
                avg_imb = surplus_trades["imb_price"].mean()
                win_rate = (surplus_trades["profit_if_surplus_trade"] > 0).mean()
                print(f"SURPLUS trades (sell IDM when surprise < -{threshold} MW):")
                print(f"  N trades:     {len(surplus_trades):,}")
                print(f"  Avg IDM:      {avg_idm:.1f} EUR")
                print(f"  Avg Imb:      {avg_imb:.1f} EUR")
                print(f"  Avg Profit:   {avg_profit:+.1f} EUR/MWh")
                print(f"  Win rate:     {win_rate*100:.0f}%")
            print()

            if len(deficit_trades) >= 10:
                avg_profit = deficit_trades["profit_if_deficit_trade"].mean()
                avg_idm = deficit_trades["idm_price"].mean()
                avg_imb = deficit_trades["imb_price"].mean()
                win_rate = (deficit_trades["profit_if_deficit_trade"] > 0).mean()
                print(f"DEFICIT trades (buy IDM when surprise > +{threshold} MW):")
                print(f"  N trades:     {len(deficit_trades):,}")
                print(f"  Avg IDM:      {avg_idm:.1f} EUR")
                print(f"  Avg Imb:      {avg_imb:.1f} EUR")
                print(f"  Avg Profit:   {avg_profit:+.1f} EUR/MWh")
                print(f"  Win rate:     {win_rate*100:.0f}%")
            print()


def main():
    print("=" * 60)
    print("IDM Trading Strategy Analysis")
    print("=" * 60)

    idm = load_idm_data()
    imb = load_imbalance_prices()
    preds = load_predictions()

    merged = analyze_strategy(idm, imb, preds)
    print_results(merged)

    print()
    print("=" * 60)
    print("[+] Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
