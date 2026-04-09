"""
03_analysis.py - Correlation, decomposition, and regime analysis.

Plot 1: Feature-target correlation heatmap (top predictors per target)
Plot 2: Decomposition-based deviations (forecast deviations vs price deviations)
Plot 3: Regime analysis (supply mix, temperature, weekday vs weekend)
Plot 4: DA-IDM spread drivers (what predicts the spread?)

Input:  da_daily_features.csv
Output: plots/01-04_*.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "MarketPriceGap" / "da_decision_matrix" / "data"
PLOT_DIR = ROOT / "MarketPriceGap" / "da_decision_matrix" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_DIR / "da_daily_features.csv", parse_dates=[0], index_col=0)
    print(f"[*] Loaded {len(df)} days, {len(df.columns)} columns")
    return df


# ==============================================================
# PLOT 1: Feature-Target Correlation Heatmap
# ==============================================================
def plot_correlation_heatmap(df):
    print("[*] Plot 1: Feature-target correlation heatmap...")

    # Define targets
    targets = [
        "da_price_mean", "da_price_std", "da_price_range",
        "spread_da_idm_mean", "spread_positive",
        "da_price_max_hour", "peak_in_evening",
        "high_volatility", "extreme_price_flag",
    ]
    targets = [t for t in targets if t in df.columns]

    # Define features (exclude targets and non-feature cols)
    exclude = set(targets) | {
        "da_price_max", "da_price_min", "da_price_min_hour",
        "da_price_median", "peak_in_morning",
        "idm_vwap_mean",  # this is a target-derived value
    }
    features = [c for c in df.columns if c not in exclude and df[c].dtype in ["float64", "int64", "int32"]]

    # Compute correlations
    corr_matrix = df[features + targets].corr()
    corr_ft = corr_matrix.loc[features, targets]

    # Drop rows with all NaN
    corr_ft = corr_ft.dropna(how="all")

    # Sort by absolute correlation to da_price_mean
    if "da_price_mean" in corr_ft.columns:
        sort_col = corr_ft["da_price_mean"].abs()
        corr_ft = corr_ft.loc[sort_col.sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(14, max(10, len(corr_ft) * 0.3)))
    im = ax.imshow(corr_ft.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([t.replace("_", "\n") for t in targets], fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(len(corr_ft)))
    ax.set_yticklabels(corr_ft.index, fontsize=7)

    # Annotate cells
    for i in range(len(corr_ft)):
        for j in range(len(targets)):
            val = corr_ft.iloc[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color)

    plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.6)
    ax.set_title("Feature-Target Correlation Matrix\n(DA Decision Matrix, Sep 2025 - Jan 2026)", fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "01_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[+]   Saved 01_correlation_heatmap.png")

    # Print top-5 features per target
    print("\n--- Top-5 Features per Target ---")
    for t in targets:
        if t in corr_ft.columns:
            top = corr_ft[t].abs().sort_values(ascending=False).head(5)
            print(f"\n  {t}:")
            for feat, val in top.items():
                sign = "+" if corr_ft.loc[feat, t] > 0 else "-"
                print(f"    {sign}{val:.3f}  {feat}")


# ==============================================================
# PLOT 2: Decomposition-Based Deviations
# ==============================================================
def plot_deviation_analysis(df):
    print("\n[*] Plot 2: Forecast deviations vs price deviations...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 2a: Load forecast deviation vs DA price
    ax = axes[0, 0]
    x = df["load_fcst_vs_7d"].dropna()
    y = df.loc[x.index, "da_price_mean"]
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    ax.scatter(x, y, alpha=0.5, s=30, c=df.loc[x.index, "month"], cmap="coolwarm")
    if len(x) > 5:
        z = np.polyfit(x, y, 1)
        r = np.corrcoef(x, y)[0, 1]
        ax.plot(sorted(x), np.polyval(z, sorted(x)), "r--", linewidth=2)
        ax.set_title(f"Load Forecast Deviation vs DA Price\nr = {r:.3f}", fontsize=10)
    ax.set_xlabel("Load Forecast vs 7d Mean (MW)")
    ax.set_ylabel("DA Price Mean (EUR/MWh)")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")

    # 2b: Temperature forecast deviation vs DA price
    ax = axes[0, 1]
    x = df["temp_fcst_vs_7d"].dropna()
    y = df.loc[x.index, "da_price_mean"]
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    ax.scatter(x, y, alpha=0.5, s=30, c=df.loc[x.index, "month"], cmap="coolwarm")
    if len(x) > 5:
        z = np.polyfit(x, y, 1)
        r = np.corrcoef(x, y)[0, 1]
        ax.plot(sorted(x), np.polyval(z, sorted(x)), "r--", linewidth=2)
        ax.set_title(f"Temperature Deviation vs DA Price\nr = {r:.3f}", fontsize=10)
    ax.set_xlabel("Temp Forecast vs 7d Mean (C)")
    ax.set_ylabel("DA Price Mean (EUR/MWh)")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")

    # 2c: Solar forecast deviation vs DA price
    ax = axes[1, 0]
    x = df["solar_fcst_vs_7d"].dropna()
    y = df.loc[x.index, "da_price_mean"]
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    ax.scatter(x, y, alpha=0.5, s=30, c=df.loc[x.index, "month"], cmap="coolwarm")
    if len(x) > 5:
        z = np.polyfit(x, y, 1)
        r = np.corrcoef(x, y)[0, 1]
        ax.plot(sorted(x), np.polyval(z, sorted(x)), "r--", linewidth=2)
        ax.set_title(f"Solar Forecast Deviation vs DA Price\nr = {r:.3f}", fontsize=10)
    ax.set_xlabel("Solar Forecast vs 7d Mean (MWh)")
    ax.set_ylabel("DA Price Mean (EUR/MWh)")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")

    # 2d: Combined deviation index
    ax = axes[1, 1]
    # Normalize deviations and combine
    cols_dev = ["load_fcst_vs_7d", "temp_fcst_vs_7d", "solar_fcst_vs_7d"]
    valid = df[cols_dev + ["da_price_mean", "month"]].dropna()
    if len(valid) > 10:
        # Z-score normalize
        for c in cols_dev:
            valid[c + "_z"] = (valid[c] - valid[c].mean()) / valid[c].std()
        # Composite: high load deviation + low temp deviation + low solar = expensive
        valid["demand_pressure"] = (
            valid["load_fcst_vs_7d_z"]
            - valid["temp_fcst_vs_7d_z"]
            - valid["solar_fcst_vs_7d_z"]
        ) / 3
        x = valid["demand_pressure"]
        y = valid["da_price_mean"]
        ax.scatter(x, y, alpha=0.5, s=30, c=valid["month"], cmap="coolwarm")
        z = np.polyfit(x, y, 1)
        r = np.corrcoef(x, y)[0, 1]
        ax.plot(sorted(x), np.polyval(z, sorted(x)), "r--", linewidth=2)
        ax.set_title(f"Combined Demand Pressure Index vs DA Price\nr = {r:.3f}", fontsize=10)
        ax.set_xlabel("Demand Pressure (z-score composite)")
        ax.set_ylabel("DA Price Mean (EUR/MWh)")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")

    fig.suptitle("Forecast Deviations vs DA Price Outcomes\n(color = month, Sep=cool, Jan=warm)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "02_deviation_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[+]   Saved 02_deviation_analysis.png")


# ==============================================================
# PLOT 3: Regime Analysis
# ==============================================================
def plot_regime_analysis(df):
    print("\n[*] Plot 3: Regime analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 3a: Load vs Price, colored by temperature
    ax = axes[0, 0]
    x = df["load_fcst_mean"]
    y = df["da_price_mean"]
    c = df["temp_fcst_mean"]
    mask = x.notna() & y.notna() & c.notna()
    sc = ax.scatter(x[mask], y[mask], c=c[mask], cmap="coolwarm_r", s=40, alpha=0.7, edgecolors="k", linewidths=0.3)
    plt.colorbar(sc, ax=ax, label="Temp Forecast (C)")
    ax.set_xlabel("Load Forecast Mean (MW)")
    ax.set_ylabel("DA Price Mean (EUR/MWh)")
    ax.set_title("Load vs Price (colored by temperature)")

    # 3b: Yesterday's Nuclear + Gas vs Price (DA-available lagged features)
    ax = axes[0, 1]
    nuc_col = "nuclear_yesterday" if "nuclear_yesterday" in df.columns else None
    gas_col = "gas_yesterday" if "gas_yesterday" in df.columns else None
    if nuc_col and gas_col:
        x = df[nuc_col]
        y = df["da_price_mean"]
        c = df[gas_col]
        mask = x.notna() & y.notna() & c.notna()
        sc = ax.scatter(x[mask], y[mask], c=c[mask], cmap="YlOrRd", s=40, alpha=0.7, edgecolors="k", linewidths=0.3)
        plt.colorbar(sc, ax=ax, label="Gas Yesterday (MW)")
        ax.set_xlabel("Nuclear Yesterday (MW)")
        ax.set_ylabel("DA Price Mean (EUR/MWh)")
        ax.set_title("Yesterday Nuclear vs Price (colored by gas)")

    # 3c: Weekday vs Weekend price distributions
    ax = axes[1, 0]
    wd = df[df["is_weekend"] == 0]["da_price_mean"].dropna()
    we = df[df["is_weekend"] == 1]["da_price_mean"].dropna()
    bins = np.linspace(df["da_price_mean"].min(), df["da_price_mean"].max(), 20)
    ax.hist(wd, bins=bins, alpha=0.6, label=f"Weekday (n={len(wd)}, mean={wd.mean():.0f})", color="steelblue")
    ax.hist(we, bins=bins, alpha=0.6, label=f"Weekend (n={len(we)}, mean={we.mean():.0f})", color="coral")
    ax.set_xlabel("DA Price Mean (EUR/MWh)")
    ax.set_ylabel("Count")
    ax.set_title("DA Price: Weekday vs Weekend")
    ax.legend(fontsize=9)

    # 3d: Solar share vs Price, by weekday/weekend
    ax = axes[1, 1]
    x = df["solar_share_of_load"]
    y = df["da_price_mean"]
    mask = x.notna() & y.notna()
    wd_mask = mask & (df["is_weekend"] == 0)
    we_mask = mask & (df["is_weekend"] == 1)
    ax.scatter(x[wd_mask], y[wd_mask], alpha=0.6, s=30, label="Weekday", marker="o", color="steelblue")
    ax.scatter(x[we_mask], y[we_mask], alpha=0.6, s=30, label="Weekend", marker="s", color="coral")
    ax.set_xlabel("Solar Share of Load (ratio)")
    ax.set_ylabel("DA Price Mean (EUR/MWh)")
    ax.set_title("Solar Penetration vs Price")
    ax.legend(fontsize=9)

    fig.suptitle("Price Regime Analysis\n(Sep 2025 - Jan 2026, 139 days)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "03_regime_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[+]   Saved 03_regime_analysis.png")


# ==============================================================
# PLOT 4: DA-IDM Spread Drivers
# ==============================================================
def plot_spread_drivers(df):
    print("\n[*] Plot 4: DA-IDM spread drivers...")

    spread = df["spread_da_idm_mean"].dropna()
    if len(spread) < 20:
        print("[-]   Not enough spread data, skipping plot 4")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 4a: Spread time series
    ax = axes[0, 0]
    colors = ["green" if s > 0 else "red" for s in spread]
    ax.bar(spread.index, spread.values, color=colors, alpha=0.7, width=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("DA - IDM Spread (EUR/MWh)")
    ax.set_title(f"DA-IDM Spread Time Series\nMean={spread.mean():.1f}, Std={spread.std():.1f}")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    # 4b: Spread by temperature bucket
    ax = axes[0, 1]
    valid = df[["temp_fcst_mean", "spread_da_idm_mean"]].dropna()
    if len(valid) > 10:
        bins = pd.qcut(valid["temp_fcst_mean"], 5, duplicates="drop")
        grouped = valid.groupby(bins)["spread_da_idm_mean"]
        means = grouped.mean()
        stds = grouped.std()
        x_labels = [f"{b.left:.0f}-{b.right:.0f}" for b in means.index]
        x_pos = range(len(means))
        ax.bar(x_pos, means.values, yerr=stds.values, capsize=4, color="steelblue", alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Temperature Forecast (C)")
        ax.set_ylabel("Mean DA-IDM Spread (EUR/MWh)")
        ax.set_title("Spread by Temperature Bucket")

    # 4c: Spread by load bucket
    ax = axes[1, 0]
    valid = df[["load_fcst_mean", "spread_da_idm_mean"]].dropna()
    if len(valid) > 10:
        bins = pd.qcut(valid["load_fcst_mean"], 5, duplicates="drop")
        grouped = valid.groupby(bins)["spread_da_idm_mean"]
        means = grouped.mean()
        stds = grouped.std()
        x_labels = [f"{b.left:.0f}-{b.right:.0f}" for b in means.index]
        x_pos = range(len(means))
        ax.bar(x_pos, means.values, yerr=stds.values, capsize=4, color="darkorange", alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Load Forecast Mean (MW)")
        ax.set_ylabel("Mean DA-IDM Spread (EUR/MWh)")
        ax.set_title("Spread by Load Bucket")

    # 4d: Spread by day-of-week
    ax = axes[1, 1]
    valid = df[["day_of_week", "spread_da_idm_mean"]].dropna()
    if len(valid) > 10:
        grouped = valid.groupby("day_of_week")["spread_da_idm_mean"]
        means = grouped.mean()
        stds = grouped.std()
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        colors_dow = ["steelblue"] * 5 + ["coral"] * 2
        ax.bar(range(7), means.values, yerr=stds.values, capsize=4, color=colors_dow, alpha=0.7)
        ax.set_xticks(range(7))
        ax.set_xticklabels(day_names)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Day of Week")
        ax.set_ylabel("Mean DA-IDM Spread (EUR/MWh)")
        ax.set_title("Spread by Day of Week")

    fig.suptitle("DA-IDM Spread Drivers\n(positive = DA more expensive than IDM)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "04_spread_drivers.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[+]   Saved 04_spread_drivers.png")

    # Print conditional spread stats
    print("\n--- Conditional Spread Statistics ---")
    valid = df[["spread_da_idm_mean", "cold_spell", "is_weekend",
                "load_fcst_vs_7d", "temp_fcst_vs_7d"]].dropna()
    if len(valid) > 10:
        # Cold spell
        cold = valid[valid["cold_spell"] == 1]["spread_da_idm_mean"]
        warm = valid[valid["cold_spell"] == 0]["spread_da_idm_mean"]
        print(f"  Cold spell (temp<0): spread = {cold.mean():+.1f} (n={len(cold)})")
        print(f"  No cold spell:       spread = {warm.mean():+.1f} (n={len(warm)})")

        # Weekend
        we = valid[valid["is_weekend"] == 1]["spread_da_idm_mean"]
        wd = valid[valid["is_weekend"] == 0]["spread_da_idm_mean"]
        print(f"  Weekend:   spread = {we.mean():+.1f} (n={len(we)})")
        print(f"  Weekday:   spread = {wd.mean():+.1f} (n={len(wd)})")

        # High load deviation
        high_load = valid[valid["load_fcst_vs_7d"] > valid["load_fcst_vs_7d"].quantile(0.75)]["spread_da_idm_mean"]
        low_load = valid[valid["load_fcst_vs_7d"] < valid["load_fcst_vs_7d"].quantile(0.25)]["spread_da_idm_mean"]
        print(f"  High load dev (Q4): spread = {high_load.mean():+.1f} (n={len(high_load)})")
        print(f"  Low load dev (Q1):  spread = {low_load.mean():+.1f} (n={len(low_load)})")


def main():
    print("=" * 70)
    print("DA Decision Matrix - Step 3: Analysis")
    print("=" * 70)

    df = load_data()
    plot_correlation_heatmap(df)
    plot_deviation_analysis(df)
    plot_regime_analysis(df)
    plot_spread_drivers(df)

    print("\n[+] All plots saved to:", PLOT_DIR)


if __name__ == "__main__":
    main()
