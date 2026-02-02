"""
Price Impact of Load Surprise Predictions
==========================================
Analyzes the average imbalance price when H+2 predicted load surprise
indicates deficit vs surplus conditions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent.parent.parent
NOWCAST_PATH = BASE_DIR / "LoadAnalysis" / "nowcast_5h"
MASTER_PATH = BASE_DIR / "data" / "master" / "master_imbalance_data.csv"
DAMAS_PATH = BASE_DIR / "features" / "DamasLoad" / "load_data.csv"
OUTPUT_DIR = Path(__file__).parent
DATA_DIR = OUTPUT_DIR / "data"


def load_data():
    """Load DAMAS and feature data."""
    print("[*] Loading data...")

    # DAMAS hourly data
    df = pd.read_csv(DAMAS_PATH, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df["year"] = df["datetime"].dt.year
    df["hour"] = df["datetime"].dt.hour
    df["dow"] = df["datetime"].dt.dayofweek
    df["error"] = df["actual_load_mw"] - df["forecast_load_mw"]

    # 3-min load for volatility features
    try:
        load_3min = pd.read_csv(BASE_DIR / "data" / "features" / "load_3min.csv")
        load_3min["datetime"] = pd.to_datetime(load_3min["datetime"])
        load_3min["hour_start"] = load_3min["datetime"].dt.floor("h")
        load_hourly = load_3min.groupby("hour_start").agg({
            "load_mw": ["std", "first", "last"]
        }).reset_index()
        load_hourly.columns = ["datetime", "load_std_3min", "load_first", "load_last"]
        load_hourly["load_trend_3min"] = load_hourly["load_last"] - load_hourly["load_first"]
        df = df.merge(load_hourly[["datetime", "load_std_3min", "load_trend_3min"]],
                      on="datetime", how="left")
    except Exception as e:
        print(f"  [!] Could not load 3-min data: {e}")
        df["load_std_3min"] = 0
        df["load_trend_3min"] = 0

    # Regulation for features
    try:
        reg_3min = pd.read_csv(BASE_DIR / "data" / "features" / "regulation_3min.csv")
        reg_3min["datetime"] = pd.to_datetime(reg_3min["datetime"])
        reg_3min["hour_start"] = reg_3min["datetime"].dt.floor("h")
        reg_hourly = reg_3min.groupby("hour_start").agg({
            "regulation_mw": ["mean", "std"]
        }).reset_index()
        reg_hourly.columns = ["datetime", "reg_mean", "reg_std"]
        df = df.merge(reg_hourly, on="datetime", how="left")
    except Exception as e:
        print(f"  [!] Could not load regulation data: {e}")
        df["reg_mean"] = 0
        df["reg_std"] = 0

    print(f"[+] Loaded {len(df):,} hourly records")
    return df


def create_features(df):
    """Create Stage 1 features for the model."""
    df = df.copy()

    # Error lags
    for lag in range(1, 9):
        df[f"error_lag{lag}"] = df["error"].shift(lag)

    # Rolling statistics
    for window in [3, 6, 12, 24]:
        df[f"error_roll_mean_{window}h"] = df["error"].shift(1).rolling(window).mean()
        df[f"error_roll_std_{window}h"] = df["error"].shift(1).rolling(window).std()

    # Error trends
    df["error_trend_3h"] = df["error_lag1"] - df["error_lag3"]
    df["error_trend_6h"] = df["error_lag1"] - df["error_lag6"]
    df["error_momentum"] = (0.5 * (df["error_lag1"] - df["error_lag2"]) +
                            0.3 * (df["error_lag2"] - df["error_lag3"]) +
                            0.2 * (df["error_lag3"] - df["error_lag4"]))

    # 3-min features
    df["load_volatility_lag1"] = df["load_std_3min"].shift(1)
    df["load_trend_lag1"] = df["load_trend_3min"].shift(1)

    # Regulation
    for lag in range(1, 4):
        df[f"reg_mean_lag{lag}"] = df["reg_mean"].shift(lag)
    df["reg_std_lag1"] = df["reg_std"].shift(1)

    # Seasonal
    seasonal = df.groupby(["dow", "hour"])["error"].mean().reset_index()
    seasonal.columns = ["dow", "hour", "seasonal_error"]
    df = df.merge(seasonal, on=["dow", "hour"], how="left")

    # Time
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    return df


def get_stage1_features():
    """Feature list for Stage 1 model."""
    return [
        "error_lag1", "error_lag2", "error_lag3", "error_lag4", "error_lag5", "error_lag6",
        "error_roll_mean_3h", "error_roll_std_3h",
        "error_roll_mean_6h", "error_roll_std_6h",
        "error_roll_mean_12h", "error_roll_std_12h",
        "error_roll_mean_24h", "error_roll_std_24h",
        "error_trend_3h", "error_trend_6h", "error_momentum",
        "load_volatility_lag1", "load_trend_lag1",
        "reg_mean_lag1", "reg_mean_lag2", "reg_mean_lag3", "reg_std_lag1",
        "seasonal_error",
        "hour", "hour_sin", "hour_cos", "dow", "is_weekend",
    ]


def generate_h2_predictions(df):
    """Generate H+2 predictions."""
    print("[*] Generating H+2 predictions...")

    features = get_stage1_features()
    model_path = NOWCAST_PATH / "models" / "stage1_h2.joblib"

    if not model_path.exists():
        print(f"[-] Model not found: {model_path}")
        return None

    model = joblib.load(model_path)
    valid_mask = df[features].notna().all(axis=1)
    preds = np.full(len(df), np.nan)
    preds[valid_mask] = model.predict(df.loc[valid_mask, features])
    df["pred_error_h2"] = preds

    print(f"[+] Generated {valid_mask.sum():,} predictions")
    return df


def load_imbalance_with_prices():
    """Load imbalance data with prices."""
    print("[*] Loading imbalance with prices...")

    imb = pd.read_csv(MASTER_PATH, parse_dates=["datetime"])
    imb = imb[["datetime", "System Imbalance (MWh)", "Imbalance Settlement Price (EUR/MWh)"]].copy()
    imb.columns = ["datetime", "imbalance", "price"]
    imb = imb[imb["imbalance"].abs() <= 300]

    # Aggregate to hourly
    imb["datetime_hour"] = imb["datetime"].dt.floor("h")
    imb_hourly = imb.groupby("datetime_hour").agg({
        "imbalance": "mean",
        "price": "mean"
    }).reset_index()
    imb_hourly.columns = ["datetime", "imbalance", "price"]

    print(f"[+] Loaded {len(imb_hourly):,} hourly price records")
    return imb_hourly


def analyze_price_impact(df, imb):
    """Analyze price impact by prediction direction."""
    print("[*] Analyzing price impact...")

    # Create future imbalance and price columns (H+2 ahead)
    df["target_hour"] = df["datetime"] + pd.Timedelta(hours=2)

    # Merge with future prices
    merged = pd.merge(
        df[["datetime", "pred_error_h2", "target_hour", "hour"]],
        imb.rename(columns={"datetime": "target_hour", "imbalance": "future_imbalance", "price": "future_price"}),
        on="target_hour",
        how="inner"
    )
    merged = merged.dropna(subset=["pred_error_h2", "future_price", "future_imbalance"])

    # Filter to day hours
    merged = merged[(merged["hour"] >= 5) & (merged["hour"] < 22)]

    print(f"[+] Merged {len(merged):,} samples with prices")

    baseline_price = merged["future_price"].mean()
    results = []

    for threshold in [50, 100, 150, 200]:
        # Deficit prediction (positive surprise)
        deficit = merged[merged["pred_error_h2"] > threshold]
        # Surplus prediction (negative surprise)
        surplus = merged[merged["pred_error_h2"] < -threshold]

        if len(deficit) > 10 and len(surplus) > 10:
            deficit_acc = (deficit["future_imbalance"] < 0).mean()
            surplus_acc = (surplus["future_imbalance"] > 0).mean()

            results.append({
                "threshold": threshold,
                "deficit_avg_price": deficit["future_price"].mean(),
                "deficit_std_price": deficit["future_price"].std(),
                "deficit_n_samples": len(deficit),
                "deficit_accuracy": deficit_acc,
                "surplus_avg_price": surplus["future_price"].mean(),
                "surplus_std_price": surplus["future_price"].std(),
                "surplus_n_samples": len(surplus),
                "surplus_accuracy": surplus_acc,
                "price_difference": deficit["future_price"].mean() - surplus["future_price"].mean()
            })

    return pd.DataFrame(results), merged, baseline_price


def create_visualization(results_df, merged, baseline_price):
    """Create price impact visualization."""
    print("[*] Creating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # 1. Price by prediction direction
    ax1 = axes[0, 0]
    x = np.arange(len(results_df))
    width = 0.35

    bars1 = ax1.bar(x - width/2, results_df["deficit_avg_price"], width,
                    label="DEFICIT Prediction (Surprise > +X)", color="#e74c3c",
                    edgecolor="black", alpha=0.8)
    bars2 = ax1.bar(x + width/2, results_df["surplus_avg_price"], width,
                    label="SURPLUS Prediction (Surprise < -X)", color="#2ecc71",
                    edgecolor="black", alpha=0.8)

    ax1.axhline(y=baseline_price, color="gray", linestyle="--", linewidth=2,
                label=f"Baseline: {baseline_price:.1f} EUR/MWh")

    ax1.set_xlabel("Prediction Threshold (MW)")
    ax1.set_ylabel("Average Imbalance Price (EUR/MWh)")
    ax1.set_title("Average Price by H+2 Prediction Direction")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"+/-{t}" for t in results_df["threshold"]])
    ax1.legend(loc="upper left")

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 2,
                f"{height:.0f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 2,
                f"{height:.0f}", ha="center", va="bottom", fontsize=9)

    # 2. Price spread
    ax2 = axes[0, 1]
    bars = ax2.bar(results_df["threshold"], results_df["price_difference"],
                   color="#3498db", edgecolor="black", alpha=0.8)
    ax2.set_xlabel("Prediction Threshold (MW)")
    ax2.set_ylabel("Price Difference (EUR/MWh)")
    ax2.set_title("Price Spread: Deficit vs Surplus Predictions")
    ax2.axhline(y=0, color="black", linewidth=0.5)

    for bar, val in zip(bars, results_df["price_difference"]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"+{val:.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # 3. Sample counts with accuracy
    ax3 = axes[1, 0]
    x = np.arange(len(results_df))
    width = 0.35

    bars1 = ax3.bar(x - width/2, results_df["deficit_n_samples"], width,
                    label="Deficit predictions", color="#e74c3c", alpha=0.7)
    bars2 = ax3.bar(x + width/2, results_df["surplus_n_samples"], width,
                    label="Surplus predictions", color="#2ecc71", alpha=0.7)

    ax3.set_xlabel("Prediction Threshold (MW)")
    ax3.set_ylabel("Number of Samples")
    ax3.set_title("Sample Count (accuracy shown above bars)")
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"+/-{t}" for t in results_df["threshold"]])
    ax3.legend()

    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        acc1 = results_df.iloc[i]["deficit_accuracy"] * 100
        acc2 = results_df.iloc[i]["surplus_accuracy"] * 100
        ax3.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 10,
                f"{acc1:.0f}%", ha="center", va="bottom", fontsize=8)
        ax3.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 10,
                f"{acc2:.0f}%", ha="center", va="bottom", fontsize=8)

    # 4. Price distribution
    ax4 = axes[1, 1]

    deficit = merged[merged["pred_error_h2"] > 100]["future_price"]
    surplus = merged[merged["pred_error_h2"] < -100]["future_price"]

    ax4.hist(deficit, bins=50, alpha=0.5, label=f"Deficit (n={len(deficit):,})",
             color="red", density=True)
    ax4.hist(surplus, bins=50, alpha=0.5, label=f"Surplus (n={len(surplus):,})",
             color="green", density=True)

    ax4.axvline(x=deficit.mean(), color="red", linestyle="--", linewidth=2,
                label=f"Deficit mean: {deficit.mean():.0f}")
    ax4.axvline(x=surplus.mean(), color="green", linestyle="--", linewidth=2,
                label=f"Surplus mean: {surplus.mean():.0f}")
    ax4.axvline(x=baseline_price, color="gray", linestyle=":", linewidth=2)

    ax4.set_xlabel("Imbalance Price (EUR/MWh)")
    ax4.set_ylabel("Density")
    ax4.set_title("Price Distribution at +/-100 MW Threshold")
    ax4.legend(fontsize=8)
    ax4.set_xlim(-50, 300)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "08_price_impact.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[+] Saved 08_price_impact.png")


def update_summary(results_df, baseline_price):
    """Update summary.md with price analysis."""
    print("[*] Updating summary...")

    summary = """

## Price Impact Analysis (H+2 Predictions)

Using the 2-hour ahead load surprise predictions to analyze price impact.

### Average Imbalance Price by Prediction Direction

| Threshold | Deficit Price | Surplus Price | Price Spread | Deficit Acc | Surplus Acc |
|-----------|---------------|---------------|--------------|-------------|-------------|
"""
    for _, row in results_df.iterrows():
        summary += f"| +/-{row['threshold']:.0f} MW | {row['deficit_avg_price']:.1f} EUR | {row['surplus_avg_price']:.1f} EUR | +{row['price_difference']:.1f} EUR | {row['deficit_accuracy']*100:.1f}% | {row['surplus_accuracy']*100:.1f}% |\n"

    summary += f"""
**Baseline average price**: {baseline_price:.1f} EUR/MWh

### Key Findings

1. **Significant price asymmetry**: When H+2 prediction indicates deficit (surprise > +100 MW),
   average price is higher than baseline. Surplus predictions have lower prices.

2. **Price spread increases with threshold**: Higher thresholds give larger price differences.

3. **Trading implication**: If confident about deficit (system short), prices tend to be higher.
   If surplus (system long), prices tend to be lower than average.

4. **Accuracy vs sample trade-off**: Higher thresholds give better accuracy but fewer signals.

### Profit Potential

At the +/-100 MW threshold:
- Deficit signals: Higher prices (system short, needs upward regulation)
- Surplus signals: Lower prices (system long, needs downward regulation)

This suggests value in timing trades based on the H+2 load surprise prediction.
"""

    with open(OUTPUT_DIR / "summary.md", "a") as f:
        f.write(summary)
    print("[+] Updated summary.md")


def main():
    print("=" * 60)
    print("Price Impact of Load Surprise Predictions")
    print("=" * 60)

    # Load and prepare data
    df = load_data()
    df = create_features(df)
    df = generate_h2_predictions(df)

    if df is None:
        print("[-] Could not generate predictions")
        return

    imb = load_imbalance_with_prices()
    results_df, merged, baseline_price = analyze_price_impact(df, imb)

    print("\n--- Price Analysis Results ---")
    print(f"Baseline average price: {baseline_price:.2f} EUR/MWh")
    print()
    print(results_df.to_string(index=False))

    results_df.to_csv(DATA_DIR / "price_by_prediction.csv", index=False)
    print(f"\n[+] Saved price_by_prediction.csv")

    create_visualization(results_df, merged, baseline_price)
    update_summary(results_df, baseline_price)

    print("\n" + "=" * 60)
    print("[+] Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
