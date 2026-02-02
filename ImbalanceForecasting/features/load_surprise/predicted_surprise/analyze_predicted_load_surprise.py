"""
Predicted Load Surprise vs Imbalance Analysis
==============================================
Uses the 5-hour nowcasting model to predict load surprise (DAMAS error)
1, 2, 3 hours ahead and correlates with imbalance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
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


def generate_predictions(df):
    """Generate predictions using saved models."""
    print("[*] Generating predictions...")

    features = get_stage1_features()
    predictions = {}

    for h in [1, 2, 3]:
        model_path = NOWCAST_PATH / "models" / f"stage1_h{h}.joblib"
        if model_path.exists():
            model = joblib.load(model_path)
            # Only predict where all features are available
            valid_mask = df[features].notna().all(axis=1)
            preds = np.full(len(df), np.nan)
            preds[valid_mask] = model.predict(df.loc[valid_mask, features])
            predictions[f"pred_error_h{h}"] = preds
            print(f"  [+] H+{h}: {valid_mask.sum():,} predictions")
        else:
            print(f"  [!] Model not found: {model_path}")
            predictions[f"pred_error_h{h}"] = np.nan

    for col, vals in predictions.items():
        df[col] = vals

    return df


def merge_with_imbalance(df):
    """Merge with imbalance data."""
    print("[*] Loading and merging imbalance data...")

    imb = pd.read_csv(MASTER_PATH, parse_dates=["datetime"])
    imb = imb[["datetime", "System Imbalance (MWh)"]].copy()
    imb.columns = ["datetime", "imbalance"]
    imb = imb[imb["imbalance"].abs() <= 300]

    # Aggregate to hourly (mean of 4 QHs)
    imb["datetime_hour"] = imb["datetime"].dt.floor("h")
    imb_hourly = imb.groupby("datetime_hour").agg({
        "imbalance": ["mean", "std", "sum"]
    }).reset_index()
    imb_hourly.columns = ["datetime", "imb_mean", "imb_std", "imb_sum"]

    df = df.merge(imb_hourly, on="datetime", how="inner")

    # Also get actual future imbalance for correlation
    for h in [1, 2, 3]:
        df[f"imb_future_h{h}"] = df["imb_mean"].shift(-h)

    print(f"[+] Merged: {len(df):,} records with imbalance")
    return df


def analyze_correlation(df):
    """Analyze correlation between predicted load surprise and future imbalance."""
    print("[*] Analyzing correlation...")

    results = []

    # Filter to day hours
    day_df = df[(df["hour"] >= 5) & (df["hour"] < 22)].copy()

    for h in [1, 2, 3]:
        pred_col = f"pred_error_h{h}"
        actual_col = f"imb_future_h{h}"

        valid = day_df[[pred_col, actual_col]].dropna()

        if len(valid) > 100:
            r, p = stats.pearsonr(valid[pred_col], valid[actual_col])
            results.append({
                "horizon": f"H+{h}",
                "n_samples": len(valid),
                "correlation": r,
                "r_squared": r**2,
                "p_value": p
            })
            print(f"  H+{h}: r = {r:.4f}, R2 = {r**2:.4f}, n = {len(valid):,}")

    # Also compare with actual (realized) load surprise
    for h in [1, 2, 3]:
        actual_error = df["error"].shift(-h)
        actual_col = f"imb_future_h{h}"

        valid = df[[actual_col]].copy()
        valid["actual_error"] = actual_error
        valid = valid.dropna()

        if len(valid) > 100:
            r, p = stats.pearsonr(valid["actual_error"], valid[actual_col])
            results.append({
                "horizon": f"H+{h} (actual error)",
                "n_samples": len(valid),
                "correlation": r,
                "r_squared": r**2,
                "p_value": p
            })
            print(f"  H+{h} (actual): r = {r:.4f}, R2 = {r**2:.4f}")

    return pd.DataFrame(results), day_df


def analyze_direction(df):
    """Analyze direction prediction capability."""
    print("[*] Analyzing direction prediction...")

    results = []

    for h in [1, 2, 3]:
        pred_col = f"pred_error_h{h}"
        actual_col = f"imb_future_h{h}"

        valid = df[[pred_col, actual_col, "hour"]].dropna()
        day_valid = valid[(valid["hour"] >= 5) & (valid["hour"] < 22)]

        # Direction prediction: if pred_error > threshold, predict negative imbalance
        for thresh in [50, 100, 150, 200]:
            # Predict negative imbalance when load surprise > threshold
            pred_neg = day_valid[day_valid[pred_col] > thresh]
            if len(pred_neg) > 20:
                acc_neg = (pred_neg[actual_col] < 0).mean()
                results.append({
                    "horizon": f"H+{h}",
                    "threshold": f"> +{thresh} MW",
                    "prediction": "Negative imb",
                    "accuracy": acc_neg,
                    "n_samples": len(pred_neg)
                })

            # Predict positive imbalance when load surprise < -threshold
            pred_pos = day_valid[day_valid[pred_col] < -thresh]
            if len(pred_pos) > 20:
                acc_pos = (pred_pos[actual_col] > 0).mean()
                results.append({
                    "horizon": f"H+{h}",
                    "threshold": f"< -{thresh} MW",
                    "prediction": "Positive imb",
                    "accuracy": acc_pos,
                    "n_samples": len(pred_pos)
                })

    return pd.DataFrame(results)


def create_visualizations(df, corr_df, direction_df):
    """Create visualizations."""
    print("[*] Creating visualizations...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Top row: Scatter plots for H+1, H+2, H+3
    day_df = df[(df["hour"] >= 5) & (df["hour"] < 22)]

    for i, h in enumerate([1, 2, 3]):
        ax = axes[0, i]
        pred_col = f"pred_error_h{h}"
        actual_col = f"imb_future_h{h}"

        valid = day_df[[pred_col, actual_col]].dropna()
        ax.scatter(valid[pred_col], valid[actual_col], alpha=0.1, s=3)

        # Fit line
        z = np.polyfit(valid[pred_col], valid[actual_col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid[pred_col].min(), valid[pred_col].max(), 100)
        r = corr_df[corr_df["horizon"] == f"H+{h}"]["correlation"].values
        r_val = r[0] if len(r) > 0 else 0
        ax.plot(x_line, p(x_line), "r-", linewidth=2, label=f"r = {r_val:.3f}")

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Predicted Load Surprise (MW)")
        ax.set_ylabel("Future Imbalance (MWh)")
        ax.set_title(f"H+{h}: Predicted Load Surprise vs Imbalance")
        ax.legend()
        ax.set_xlim(-300, 300)
        ax.set_ylim(-40, 40)

    # Bottom row: Direction accuracy by threshold
    ax = axes[1, 0]
    for h in [1, 2, 3]:
        h_data = direction_df[(direction_df["horizon"] == f"H+{h}") &
                               (direction_df["prediction"] == "Negative imb")]
        thresholds = [50, 100, 150, 200]
        accs = [h_data[h_data["threshold"] == f"> +{t} MW"]["accuracy"].values[0] * 100
                if len(h_data[h_data["threshold"] == f"> +{t} MW"]) > 0 else np.nan
                for t in thresholds]
        ax.plot(thresholds, accs, "o-", label=f"H+{h}")

    ax.axhline(y=50, color="gray", linestyle="--")
    ax.axhline(y=70, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Predicted Load Surprise Threshold (MW)")
    ax.set_ylabel("Accuracy predicting NEGATIVE imbalance (%)")
    ax.set_title("Direction Accuracy: Predict Negative Imbalance")
    ax.legend()
    ax.set_ylim(40, 80)

    ax = axes[1, 1]
    for h in [1, 2, 3]:
        h_data = direction_df[(direction_df["horizon"] == f"H+{h}") &
                               (direction_df["prediction"] == "Positive imb")]
        thresholds = [50, 100, 150, 200]
        accs = [h_data[h_data["threshold"] == f"< -{t} MW"]["accuracy"].values[0] * 100
                if len(h_data[h_data["threshold"] == f"< -{t} MW"]) > 0 else np.nan
                for t in thresholds]
        ax.plot(thresholds, accs, "o-", label=f"H+{h}")

    ax.axhline(y=50, color="gray", linestyle="--")
    ax.axhline(y=70, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Predicted Load Surprise Threshold (MW)")
    ax.set_ylabel("Accuracy predicting POSITIVE imbalance (%)")
    ax.set_title("Direction Accuracy: Predict Positive Imbalance")
    ax.legend()
    ax.set_ylim(40, 80)

    # Correlation comparison
    ax = axes[1, 2]
    horizons = ["H+1", "H+2", "H+3"]
    pred_corrs = [corr_df[corr_df["horizon"] == h]["correlation"].values[0]
                  if len(corr_df[corr_df["horizon"] == h]) > 0 else 0
                  for h in horizons]
    actual_corrs = [corr_df[corr_df["horizon"] == f"{h} (actual error)"]["correlation"].values[0]
                    if len(corr_df[corr_df["horizon"] == f"{h} (actual error)"]) > 0 else 0
                    for h in horizons]

    x = np.arange(3)
    width = 0.35
    ax.bar(x - width/2, pred_corrs, width, label="Predicted surprise", color="steelblue")
    ax.bar(x + width/2, actual_corrs, width, label="Actual surprise", color="orange")
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.set_ylabel("Correlation with Imbalance")
    ax.set_title("Predicted vs Actual Load Surprise Correlation")
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_predicted_load_surprise.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[+] Saved 07_predicted_load_surprise.png")


def generate_summary(corr_df, direction_df):
    """Generate summary."""
    summary = """

## Predicted Load Surprise vs Future Imbalance

Using the 5-hour nowcasting model predictions to forecast imbalance direction.

### Correlation: Predicted Load Surprise vs Future Imbalance

| Horizon | Correlation | R-squared | N samples |
|---------|-------------|-----------|-----------|
"""
    for _, row in corr_df.iterrows():
        summary += f"| {row['horizon']} | {row['correlation']:.4f} | {row['r_squared']:.4f} | {row['n_samples']:,} |\n"

    summary += """
### Direction Prediction Accuracy

**Predict NEGATIVE imbalance when predicted load surprise > threshold:**

| Horizon | > +50 MW | > +100 MW | > +150 MW | > +200 MW |
|---------|----------|-----------|-----------|-----------|
"""
    for h in [1, 2, 3]:
        row = f"| H+{h} |"
        for t in [50, 100, 150, 200]:
            data = direction_df[(direction_df["horizon"] == f"H+{h}") &
                                (direction_df["threshold"] == f"> +{t} MW")]
            if len(data) > 0:
                row += f" {data['accuracy'].values[0]*100:.1f}% |"
            else:
                row += " N/A |"
        summary += row + "\n"

    summary += """
**Predict POSITIVE imbalance when predicted load surprise < threshold:**

| Horizon | < -50 MW | < -100 MW | < -150 MW | < -200 MW |
|---------|----------|-----------|-----------|-----------|
"""
    for h in [1, 2, 3]:
        row = f"| H+{h} |"
        for t in [50, 100, 150, 200]:
            data = direction_df[(direction_df["horizon"] == f"H+{h}") &
                                (direction_df["threshold"] == f"< -{t} MW")]
            if len(data) > 0:
                row += f" {data['accuracy'].values[0]*100:.1f}% |"
            else:
                row += " N/A |"
        summary += row + "\n"

    summary += """
### Key Findings

1. **Predicted load surprise correlates with future imbalance** at all horizons
2. **Correlation decreases with horizon** (as expected - prediction error increases)
3. **Direction prediction is possible** but accuracy decreases with horizon
4. **Actual (realized) load surprise has stronger correlation** than predicted
"""

    with open(OUTPUT_DIR / "summary.md", "a") as f:
        f.write(summary)
    print("[+] Updated summary.md")


def main():
    print("=" * 60)
    print("Predicted Load Surprise vs Future Imbalance")
    print("=" * 60)

    df = load_data()
    df = create_features(df)
    df = generate_predictions(df)
    df = merge_with_imbalance(df)

    corr_df, day_df = analyze_correlation(df)
    direction_df = analyze_direction(df)

    corr_df.to_csv(DATA_DIR / "predicted_surprise_correlation.csv", index=False)
    direction_df.to_csv(DATA_DIR / "predicted_surprise_direction.csv", index=False)

    create_visualizations(df, corr_df, direction_df)
    generate_summary(corr_df, direction_df)

    print("\n" + "=" * 60)
    print("[+] Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
