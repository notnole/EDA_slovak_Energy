"""
IDM Strategy Visualization (Sep-Dec 2025)
==========================================
Shows how the strategy works and its performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent.parent.parent
NOWCAST_PATH = BASE_DIR / "LoadAnalysis" / "nowcast_5h"
MASTER_PATH = BASE_DIR / "data" / "master" / "master_imbalance_data.csv"
DAMAS_PATH = BASE_DIR / "features" / "DamasLoad" / "load_data.csv"
IDM_PATH = BASE_DIR / "RawData" / "IDM_MarketData"
OUTPUT_DIR = Path(__file__).parent


def load_and_prepare_data():
    """Load all data and prepare for analysis."""
    print("[*] Loading data...")

    # Load IDM data
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

    # Load predictions (simplified - reuse the feature engineering)
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
    merged = pd.merge(idm[["datetime", "date", "hour", "qh_in_hour", "idm_price"]],
                      imb, on="datetime", how="inner")
    merged["pred_hour"] = merged["datetime"].dt.floor("h")
    preds_hourly = df[["target_hour", "pred_error_h2", "year", "month"]].copy()
    preds_hourly["pred_hour"] = preds_hourly["target_hour"]
    merged = pd.merge(merged, preds_hourly[["pred_hour", "pred_error_h2", "year", "month"]],
                      on="pred_hour", how="inner")
    merged = merged.dropna(subset=["idm_price", "imb_price", "pred_error_h2"])
    merged = merged[(merged["hour"] >= 5) & (merged["hour"] < 22)]

    # Sep-Dec 2025 only
    merged = merged[(merged["year"] == 2025) & (merged["month"] >= 9)].copy()
    merged = merged.sort_values("datetime").reset_index(drop=True)

    print(f"[+] Prepared {len(merged):,} samples for Sep-Dec 2025")
    return merged


def create_visualization(merged):
    """Create comprehensive visualization."""
    print("[*] Creating visualization...")

    fig = plt.figure(figsize=(16, 12))

    # Define threshold and trade size
    THRESHOLD = 50
    TRADE_SIZE_MWH = 5  # 5 MWh per trade

    # Identify trades
    merged["is_surplus_trade"] = merged["pred_error_h2"] < -THRESHOLD
    merged["profit_per_mwh"] = np.where(merged["is_surplus_trade"],
                                 merged["idm_price"] - merged["imb_price"],
                                 0)
    merged["profit"] = merged["profit_per_mwh"] * TRADE_SIZE_MWH  # EUR profit

    # ============ Plot 1: How it works - Price comparison ============
    ax1 = fig.add_subplot(2, 2, 1)

    # Bin by predicted surprise
    bins = [-300, -150, -100, -50, 0, 50, 100, 150, 300]
    merged["surprise_bin"] = pd.cut(merged["pred_error_h2"], bins=bins)

    bin_stats = merged.groupby("surprise_bin", observed=True).agg({
        "idm_price": "mean",
        "imb_price": "mean",
        "pred_error_h2": "count"
    }).reset_index()
    bin_stats.columns = ["bin", "idm_avg", "imb_avg", "count"]
    bin_stats["bin_mid"] = bin_stats["bin"].apply(lambda x: (x.left + x.right) / 2)
    bin_stats = bin_stats.sort_values("bin_mid")

    x = np.arange(len(bin_stats))
    width = 0.35

    bars1 = ax1.bar(x - width/2, bin_stats["idm_avg"], width, label="IDM Price", color="#3498db", alpha=0.8)
    bars2 = ax1.bar(x + width/2, bin_stats["imb_avg"], width, label="Imbalance Price", color="#e74c3c", alpha=0.8)

    # Highlight the trading zone
    ax1.axvspan(-0.5, 1.5, alpha=0.2, color="green", label="SURPLUS trade zone")

    ax1.set_xlabel("Predicted Load Surprise (MW)")
    ax1.set_ylabel("Average Price (EUR/MWh)")
    ax1.set_title("How It Works: IDM vs Imbalance Price by Predicted Surprise")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{int(b.left)} to {int(b.right)}" for b in bin_stats["bin"]], rotation=45, ha="right")
    ax1.legend(loc="upper left")

    # Add spread annotation
    for i, (idm, imb) in enumerate(zip(bin_stats["idm_avg"], bin_stats["imb_avg"])):
        spread = idm - imb
        color = "green" if spread > 0 else "red"
        ax1.annotate(f"{spread:+.0f}", xy=(i, max(idm, imb) + 5), ha="center", fontsize=8, color=color)

    # ============ Plot 2: Cumulative profit over time ============
    ax2 = fig.add_subplot(2, 2, 2)

    trades = merged[merged["is_surplus_trade"]].copy()
    trades["cumulative_profit"] = trades["profit"].cumsum()
    trades["trade_num"] = range(1, len(trades) + 1)

    ax2.plot(trades["datetime"], trades["cumulative_profit"], "g-", linewidth=2)
    ax2.fill_between(trades["datetime"], 0, trades["cumulative_profit"], alpha=0.3, color="green")
    ax2.axhline(y=0, color="black", linewidth=0.5)

    ax2.set_xlabel("Date")
    ax2.set_ylabel("Cumulative Profit (EUR)")
    ax2.set_title(f"Strategy Performance: {TRADE_SIZE_MWH} MWh per trade (n={len(trades)} trades)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    total_profit = trades["profit"].sum()
    avg_profit = trades["profit"].mean()
    avg_profit_per_mwh = trades["profit_per_mwh"].mean()
    ax2.text(0.02, 0.98, f"Total: {total_profit:,.0f} EUR\nAvg per trade: {avg_profit:+.0f} EUR ({avg_profit_per_mwh:+.1f} EUR/MWh)",
             transform=ax2.transAxes, va="top", fontsize=10,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # ============ Plot 3: Individual trade profits ============
    ax3 = fig.add_subplot(2, 2, 3)

    colors = ["green" if p > 0 else "red" for p in trades["profit"]]
    ax3.bar(range(len(trades)), trades["profit"], color=colors, alpha=0.7, width=1.0)
    ax3.axhline(y=0, color="black", linewidth=1)
    ax3.axhline(y=avg_profit, color="blue", linestyle="--", linewidth=2, label=f"Avg: {avg_profit:+.0f} EUR")

    win_rate = (trades["profit"] > 0).mean()
    ax3.set_xlabel("Trade Number")
    ax3.set_ylabel(f"Profit (EUR) @ {TRADE_SIZE_MWH} MWh")
    ax3.set_title(f"Individual Trade Profits (Win Rate: {win_rate*100:.0f}%)")
    ax3.legend()

    # ============ Plot 4: Signal distribution and trade frequency ============
    ax4 = fig.add_subplot(2, 2, 4)

    # Daily trade count
    trades["trade_date"] = trades["datetime"].dt.date
    daily_trades = trades.groupby("trade_date").size()
    daily_profit = trades.groupby("trade_date")["profit"].sum()

    ax4_twin = ax4.twinx()

    ax4.bar(range(len(daily_trades)), daily_trades.values, alpha=0.5, color="steelblue", label="# Trades")
    ax4_twin.plot(range(len(daily_profit)), daily_profit.values, "g-o", markersize=3, label="Daily Profit")
    ax4_twin.axhline(y=0, color="black", linewidth=0.5)

    ax4.set_xlabel("Day")
    ax4.set_ylabel("Number of Trades", color="steelblue")
    ax4_twin.set_ylabel(f"Daily Profit (EUR) @ {TRADE_SIZE_MWH} MWh", color="green")
    ax4.set_title("Daily Trading Activity and Profit")

    # Stats box
    stats_text = f"Period: Sep-Dec 2025\n"
    stats_text += f"Threshold: < -{THRESHOLD} MW\n"
    stats_text += f"Total trades: {len(trades)}\n"
    stats_text += f"Trading days: {len(daily_trades)}\n"
    stats_text += f"Avg trades/day: {len(trades)/len(daily_trades):.1f}\n"
    stats_text += f"Win rate: {win_rate*100:.0f}%"

    ax4.text(0.98, 0.98, stats_text, transform=ax4.transAxes, va="top", ha="right",
             fontsize=9, bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "09_idm_strategy_sep_dec.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[+] Saved 09_idm_strategy_sep_dec.png")


def main():
    print("=" * 60)
    print("IDM Strategy Visualization (Sep-Dec 2025)")
    print("=" * 60)

    merged = load_and_prepare_data()
    create_visualization(merged)

    print("\n" + "=" * 60)
    print("[+] Visualization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
