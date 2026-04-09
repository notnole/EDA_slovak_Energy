"""
Analyze relationship between Hub-to-Hub cross-border capacity and system imbalance volume.

Hypothesis: When cross-border import capacity is constrained (congested),
the system is more likely to experience large imbalances because the grid
has fewer options to balance supply/demand mismatches via imports.

Sources:
  - H2H capacity: data/h2h_idm_hourly.csv (from extract_h2h_idm_data.py)
  - Imbalance: data/master/master_imbalance_data.csv (15-min, aggregated to hourly)

Output: 5 PNG plots + console stats + summary update
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
H2H_PATH = DATA_DIR / "h2h_idm_hourly.csv"
REPO_ROOT = SCRIPT_DIR.parents[3]
IMBALANCE_PATH = REPO_ROOT / "data" / "master" / "master_imbalance_data.csv"

# --- Config ---
AREAS = ["ceps", "apg", "mavir", "pse", "de"]
AREA_LABELS = {
    "ceps": "CEPS (CZ)",
    "apg": "APG (AT)",
    "mavir": "MAVIR (HU)",
    "pse": "PSE (PL)",
    "de": "DE (avg)",
}
IMB_CLIP = (-200, 200)  # MWh clip for visual clarity


def load_and_merge():
    """Load H2H data and imbalance data, merge at hourly resolution."""
    print("[*] Loading H2H capacity data...")
    h2h = pd.read_csv(H2H_PATH, parse_dates=["tradeday"])
    print(f"    H2H: {len(h2h)} rows, {h2h['tradeday'].min().date()} to {h2h['tradeday'].max().date()}")

    print("[*] Loading imbalance data (15-min)...")
    imb_raw = pd.read_csv(IMBALANCE_PATH, parse_dates=["datetime"])
    print(f"    Imbalance raw: {len(imb_raw)} rows, {imb_raw['datetime'].min()} to {imb_raw['datetime'].max()}")

    # Aggregate imbalance to hourly
    imb_raw["date"] = imb_raw["datetime"].dt.date
    imb_raw["hour"] = imb_raw["datetime"].dt.hour
    imb_col = "System Imbalance (MWh)"

    imb_hourly = imb_raw.groupby(["date", "hour"]).agg(
        imbalance_mwh=(imb_col, "sum"),         # Sum of 4 x 15-min
        imbalance_abs_mean=(imb_col, lambda x: x.abs().mean()),  # Mean absolute
        imbalance_std=(imb_col, "std"),          # Within-hour variability
        imbalance_min=(imb_col, "min"),          # Most negative QH
        imbalance_max=(imb_col, "max"),          # Most positive QH
        n_periods=(imb_col, "count"),            # Should be 4
    ).reset_index()

    # Sign-based features
    imb_hourly["imbalance_sign"] = np.sign(imb_hourly["imbalance_mwh"])
    imb_hourly["abs_imbalance"] = imb_hourly["imbalance_mwh"].abs()
    imb_hourly["is_deficit"] = (imb_hourly["imbalance_mwh"] < 0).astype(int)

    print(f"    Imbalance hourly: {len(imb_hourly)} rows")

    # Merge
    h2h["date"] = h2h["tradeday"].dt.date
    h2h["hour"] = h2h["hour"].astype(int)

    merged = h2h.merge(imb_hourly, on=["date", "hour"], how="inner")
    print(f"[+] Merged: {len(merged)} rows with both H2H and imbalance data")
    print(f"    Date range: {pd.to_datetime(merged['date']).min().date()} to "
          f"{pd.to_datetime(merged['date']).max().date()}")
    print(f"    Imbalance stats: mean={merged['imbalance_mwh'].mean():.1f}, "
          f"std={merged['imbalance_mwh'].std():.1f}, "
          f"min={merged['imbalance_mwh'].min():.1f}, max={merged['imbalance_mwh'].max():.1f}")
    print(f"    Deficit hours: {merged['is_deficit'].sum()} ({merged['is_deficit'].mean()*100:.1f}%)")

    return merged


def plot_correlation_matrix(df):
    """Plot 1: Correlation heatmap - H2H features vs imbalance metrics."""
    print("[*] Plotting correlation matrix...")

    cap_cols = []
    for area in AREAS:
        cap_cols.append(f"{area}_in")
    agg_cols = ["total_import_cap", "total_export_cap", "net_capacity", "congested_in_count"]
    imb_cols = ["imbalance_mwh", "abs_imbalance", "is_deficit"]

    all_cols = cap_cols + agg_cols + imb_cols
    available = [c for c in all_cols if c in df.columns]
    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    labels = [c.replace("_", " ").title() for c in available]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(len(available)):
        for j in range(len(available)):
            val = corr.iloc[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7)

    plt.colorbar(im, ax=ax, label="Pearson Correlation", shrink=0.8)
    ax.set_title("H2H Cross-Border Capacity vs System Imbalance - Correlation Matrix", fontsize=12)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "06_h2h_imbalance_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[+] Saved 06_h2h_imbalance_correlation.png")


def plot_capacity_vs_imbalance(df):
    """Plot 2: Scatter of import capacity vs imbalance per area."""
    print("[*] Plotting capacity vs imbalance scatters...")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    df_clip = df.copy()
    df_clip["imb_clip"] = df_clip["imbalance_mwh"].clip(*IMB_CLIP)

    for idx, area in enumerate(AREAS):
        ax = axes[idx]
        col = f"{area}_in"
        if col not in df_clip.columns:
            continue

        x = df_clip[col].values
        y = df_clip["imb_clip"].values
        mask = np.isfinite(x) & np.isfinite(y)

        ax.scatter(x[mask], y[mask], alpha=0.25, s=10, c="#E53935", edgecolors="none")
        ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.set_xlabel(f"{AREA_LABELS[area]} Import Cap [MW]")
        ax.set_ylabel("System Imbalance [MWh]")
        ax.set_title(AREA_LABELS[area])

        r, p = stats.pearsonr(x[mask], y[mask])
        rs, ps = stats.spearmanr(x[mask], y[mask])
        ax.text(0.05, 0.95, f"Pearson r = {r:.3f}\nSpearman r = {rs:.3f}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Total import capacity
    ax = axes[5]
    x = df_clip["total_import_cap"].values
    y = df_clip["imb_clip"].values
    mask = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[mask], y[mask], alpha=0.25, s=10, c="#1976D2", edgecolors="none")
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    r, p = stats.pearsonr(x[mask], y[mask])
    rs, ps = stats.spearmanr(x[mask], y[mask])
    ax.text(0.05, 0.95, f"Pearson r = {r:.3f}\nSpearman r = {rs:.3f}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.set_xlabel("Total Import Capacity [MW]")
    ax.set_ylabel("System Imbalance [MWh]")
    ax.set_title("Total (All Borders)")

    plt.suptitle("System Imbalance vs Cross-Border Import Capacity", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "07_h2h_imbalance_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[+] Saved 07_h2h_imbalance_scatter.png")


def plot_congestion_imbalance(df):
    """Plot 3: Imbalance distribution when borders congested vs open."""
    print("[*] Plotting congestion vs imbalance...")

    df_clip = df.copy()
    df_clip["imb_clip"] = df_clip["imbalance_mwh"].clip(*IMB_CLIP)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, area in enumerate(AREAS):
        ax = axes[idx]
        cong_col = f"{area}_congested_in"
        if cong_col not in df_clip.columns:
            continue

        open_imb = df_clip[df_clip[cong_col] == 0]["imb_clip"]
        cong_imb = df_clip[df_clip[cong_col] == 1]["imb_clip"]

        bp = ax.boxplot([open_imb, cong_imb], tick_labels=["Open", "Congested"],
                        patch_artist=True, widths=0.6,
                        medianprops=dict(color="black", linewidth=2))
        bp["boxes"][0].set_facecolor("#4CAF50")
        bp["boxes"][1].set_facecolor("#F44336")

        # Stats
        deficit_open = (df_clip[df_clip[cong_col] == 0]["is_deficit"].mean() * 100)
        deficit_cong = (df_clip[df_clip[cong_col] == 1]["is_deficit"].mean() * 100)
        abs_diff = cong_imb.abs().median() - open_imb.abs().median()

        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title(f"{AREA_LABELS[area]}\nDeficit: {deficit_open:.0f}% open, {deficit_cong:.0f}% cong")
        ax.set_ylabel("System Imbalance [MWh]")
        ax.text(0.5, -0.12, f"n={len(open_imb)}", transform=ax.transAxes, ha="left", fontsize=8, color="gray")
        ax.text(0.85, -0.12, f"n={len(cong_imb)}", transform=ax.transAxes, ha="left", fontsize=8, color="gray")

    # Last: congested count vs |imbalance|
    ax = axes[5]
    groups = df_clip.groupby("congested_in_count")
    data = []
    labels = []
    for count, grp in groups:
        data.append(grp["abs_imbalance"].clip(0, 200).values)
        labels.append(f"{int(count)}\n(n={len(grp)})")
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6,
                    medianprops=dict(color="black", linewidth=2))
    cmap = plt.cm.RdYlGn_r
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(cmap(i / max(len(bp["boxes"]) - 1, 1)))
    ax.set_title("# Congested Borders vs |Imbalance|")
    ax.set_ylabel("|System Imbalance| [MWh]")
    ax.set_xlabel("Number of congested import borders")

    plt.suptitle("System Imbalance by Border Congestion State", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "08_h2h_imbalance_congestion.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[+] Saved 08_h2h_imbalance_congestion.png")


def plot_timeseries_overlay(df):
    """Plot 4: Time series of imbalance + capacity overlay."""
    print("[*] Plotting time series overlay...")

    df_ts = df.copy()
    df_ts["datetime"] = pd.to_datetime(df_ts["date"]) + pd.to_timedelta(df_ts["hour"], unit="h")

    # Pick a representative week
    week_start = pd.Timestamp("2026-01-05")
    week_end = pd.Timestamp("2026-01-12")
    week = df_ts[(df_ts["datetime"] >= week_start) & (df_ts["datetime"] < week_end)].copy()

    if len(week) < 48:
        # Fallback
        for start in pd.date_range(df_ts["datetime"].min(), df_ts["datetime"].max() - pd.Timedelta(days=7)):
            end = start + pd.Timedelta(days=7)
            week = df_ts[(df_ts["datetime"] >= start) & (df_ts["datetime"] < end)]
            if len(week) >= 100:
                week_start, week_end = start, end
                break

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10), sharex=True,
                                          gridspec_kw={"height_ratios": [2, 2, 1.5]})

    # Top: System imbalance
    colors_imb = ["#E53935" if v < 0 else "#43A047" for v in week["imbalance_mwh"]]
    ax1.bar(week["datetime"], week["imbalance_mwh"], width=1/24, color=colors_imb, alpha=0.7)
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.set_ylabel("System Imbalance [MWh]")
    ax1.set_title(f"System Imbalance & Cross-Border Import Capacity ({week_start.date()} to {week_end.date()})")
    ax1.grid(True, alpha=0.3)

    # Middle: Import capacity stacked area
    area_colors = {"ceps": "#E53935", "apg": "#43A047", "mavir": "#FB8C00",
                   "pse": "#1E88E5", "de": "#8E24AA"}
    bottom = np.zeros(len(week))
    for area in AREAS:
        col = f"{area}_in"
        if col in week.columns:
            vals = week[col].fillna(0).values
            ax2.fill_between(week["datetime"], bottom, bottom + vals,
                             alpha=0.7, label=AREA_LABELS[area], color=area_colors[area])
            bottom += vals
    ax2.set_ylabel("Import Capacity [MW]")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Bottom: Congested border count
    ax3.fill_between(week["datetime"], 0, week["congested_in_count"],
                     alpha=0.7, color="#FF6F00", step="mid")
    ax3.set_ylabel("# Congested\nBorders")
    ax3.set_xlabel("Date")
    ax3.set_ylim(0, 5.5)
    ax3.set_yticks(range(6))
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(BASE_DIR / "09_h2h_imbalance_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[+] Saved 09_h2h_imbalance_timeseries.png")


def plot_deficit_analysis(df):
    """Plot 5: Deficit probability and magnitude by capacity state."""
    print("[*] Plotting deficit analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Deficit probability by congested count
    ax = axes[0, 0]
    deficit_by_cong = df.groupby("congested_in_count").agg(
        deficit_pct=("is_deficit", "mean"),
        n=("is_deficit", "count"),
    ).reset_index()
    bars = ax.bar(deficit_by_cong["congested_in_count"], deficit_by_cong["deficit_pct"] * 100,
                  color=plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(deficit_by_cong))),
                  edgecolor="black", linewidth=0.5)
    for bar, row in zip(bars, deficit_by_cong.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"n={row.n}", ha="center", fontsize=8, color="gray")
    ax.set_xlabel("# Congested Import Borders")
    ax.set_ylabel("Deficit Probability [%]")
    ax.set_title("System Deficit Probability by Congestion")
    ax.grid(True, alpha=0.3, axis="y")

    # Top-right: |Imbalance| by congested count
    ax = axes[0, 1]
    abs_by_cong = df.groupby("congested_in_count").agg(
        abs_mean=("abs_imbalance", "mean"),
        abs_median=("abs_imbalance", "median"),
        abs_p90=("abs_imbalance", lambda x: x.quantile(0.9)),
    ).reset_index()
    x_pos = abs_by_cong["congested_in_count"]
    ax.bar(x_pos - 0.2, abs_by_cong["abs_mean"], width=0.2, label="Mean", color="#1976D2", alpha=0.8)
    ax.bar(x_pos, abs_by_cong["abs_median"], width=0.2, label="Median", color="#43A047", alpha=0.8)
    ax.bar(x_pos + 0.2, abs_by_cong["abs_p90"], width=0.2, label="P90", color="#E53935", alpha=0.8)
    ax.set_xlabel("# Congested Import Borders")
    ax.set_ylabel("|System Imbalance| [MWh]")
    ax.set_title("Imbalance Magnitude by Congestion")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Bottom-left: Hourly profile of imbalance and congestion
    ax = axes[1, 0]
    hourly = df.groupby("hour").agg(
        imb_mean=("imbalance_mwh", "mean"),
        imb_std=("imbalance_mwh", "std"),
        abs_mean=("abs_imbalance", "mean"),
        deficit_pct=("is_deficit", "mean"),
        import_cap_mean=("total_import_cap", "mean"),
    ).reset_index()

    ax2r = ax.twinx()
    ax.bar(hourly["hour"], hourly["imb_mean"], alpha=0.5, color="#1976D2", label="Mean Imbalance")
    ax.fill_between(hourly["hour"],
                     hourly["imb_mean"] - hourly["imb_std"],
                     hourly["imb_mean"] + hourly["imb_std"],
                     alpha=0.15, color="#1976D2")
    ax2r.plot(hourly["hour"], hourly["import_cap_mean"], "r--s", markersize=4, label="Import Cap")
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Imbalance [MWh]", color="#1976D2")
    ax2r.set_ylabel("Import Capacity [MW]", color="red")
    ax.set_title("Hourly Profile: Imbalance & Import Capacity")
    ax.set_xticks(range(0, 24, 2))
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-right: Imbalance sign vs total import capacity
    ax = axes[1, 1]
    cap_bins = pd.qcut(df["total_import_cap"], q=5, duplicates="drop")
    binned = df.groupby(cap_bins, observed=True).agg(
        deficit_pct=("is_deficit", "mean"),
        surplus_pct=("is_deficit", lambda x: 1 - x.mean()),
        abs_mean=("abs_imbalance", "mean"),
        n=("is_deficit", "count"),
    ).reset_index()
    bin_labels = [f"{int(b.left)}-{int(b.right)}" for b in binned["total_import_cap"]]

    ax.bar(range(len(binned)), binned["deficit_pct"] * 100, color="#E53935", alpha=0.7, label="Deficit %")
    ax.bar(range(len(binned)), -binned["surplus_pct"] * 100, color="#43A047", alpha=0.7, label="Surplus %")
    ax.set_xticks(range(len(binned)))
    ax.set_xticklabels(bin_labels, rotation=30, fontsize=8)
    ax.set_xlabel("Total Import Capacity [MW] (quintiles)")
    ax.set_ylabel("Deficit (-) / Surplus (+) %")
    ax.set_title("Deficit vs Surplus by Import Capacity")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("System Imbalance Analysis by Cross-Border Capacity", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "10_h2h_imbalance_deficit.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[+] Saved 10_h2h_imbalance_deficit.png")


def compute_statistics(df):
    """Compute and print comprehensive statistics."""
    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS: H2H Capacity vs System Imbalance")
    print("=" * 60)

    # --- Correlations: capacity vs imbalance volume ---
    print("\n--- Correlations: Import Capacity vs Imbalance Volume ---")
    results = []
    for area in AREAS:
        col = f"{area}_in"
        if col not in df.columns:
            continue
        valid = df[[col, "imbalance_mwh"]].dropna()
        r, p = stats.pearsonr(valid[col], valid["imbalance_mwh"])
        rs, ps = stats.spearmanr(valid[col], valid["imbalance_mwh"])
        print(f"  {AREA_LABELS[area]:15s} import: Pearson r={r:+.3f} (p={p:.1e}), "
              f"Spearman r={rs:+.3f} (p={ps:.1e})")
        results.append({"area": area, "pearson_r": r, "pearson_p": p,
                        "spearman_r": rs, "spearman_p": ps})

    for col, label in [("total_import_cap", "Total Import"),
                       ("net_capacity", "Net Capacity"),
                       ("congested_in_count", "# Congested")]:
        valid = df[[col, "imbalance_mwh"]].dropna()
        r, p = stats.pearsonr(valid[col], valid["imbalance_mwh"])
        rs, ps = stats.spearmanr(valid[col], valid["imbalance_mwh"])
        print(f"  {label:15s}:       Pearson r={r:+.3f} (p={p:.1e}), "
              f"Spearman r={rs:+.3f} (p={ps:.1e})")

    # --- Correlations: capacity vs |imbalance| ---
    print("\n--- Correlations: Import Capacity vs |Imbalance| (magnitude) ---")
    for col, label in [("total_import_cap", "Total Import"),
                       ("congested_in_count", "# Congested")]:
        valid = df[[col, "abs_imbalance"]].dropna()
        r, p = stats.pearsonr(valid[col], valid["abs_imbalance"])
        rs, ps = stats.spearmanr(valid[col], valid["abs_imbalance"])
        print(f"  {label:15s}: Pearson r={r:+.3f} (p={p:.1e}), "
              f"Spearman r={rs:+.3f} (p={ps:.1e})")

    # --- Congestion vs deficit rate ---
    print("\n--- Deficit Rate by Congestion State ---")
    cong_stats = []
    for area in AREAS:
        cong_col = f"{area}_congested_in"
        if cong_col not in df.columns:
            continue
        open_df = df[df[cong_col] == 0]
        cong_df = df[df[cong_col] == 1]
        deficit_open = open_df["is_deficit"].mean() * 100
        deficit_cong = cong_df["is_deficit"].mean() * 100
        abs_open = open_df["abs_imbalance"].median()
        abs_cong = cong_df["abs_imbalance"].median()

        print(f"  {AREA_LABELS[area]:15s}: Open deficit={deficit_open:.1f}%, "
              f"Cong deficit={deficit_cong:.1f}%, "
              f"|Imb| open={abs_open:.1f}, cong={abs_cong:.1f} MWh")
        cong_stats.append({
            "area": AREA_LABELS[area],
            "deficit_open": deficit_open,
            "deficit_cong": deficit_cong,
            "abs_open": abs_open,
            "abs_cong": abs_cong,
        })

    # --- Congested count vs imbalance magnitude ---
    print("\n--- |Imbalance| by Number of Congested Borders ---")
    for count, grp in df.groupby("congested_in_count"):
        print(f"  {int(count)} borders: "
              f"mean={grp['imbalance_mwh'].mean():+.1f}, "
              f"|mean|={grp['abs_imbalance'].mean():.1f}, "
              f"|median|={grp['abs_imbalance'].median():.1f}, "
              f"|P90|={grp['abs_imbalance'].quantile(0.9):.1f}, "
              f"deficit={grp['is_deficit'].mean()*100:.1f}%, "
              f"n={len(grp)}")

    # --- Direction shift test ---
    print("\n--- Imbalance Direction by Capacity Terciles ---")
    terciles = pd.qcut(df["total_import_cap"], q=3, labels=["Low", "Mid", "High"])
    for t, grp in df.groupby(terciles, observed=True):
        print(f"  {t:5s} import cap: "
              f"mean imb={grp['imbalance_mwh'].mean():+.1f}, "
              f"deficit={grp['is_deficit'].mean()*100:.1f}%, "
              f"|imb|={grp['abs_imbalance'].mean():.1f}, "
              f"n={len(grp)}")

    return results, cong_stats


def update_summary(df, corr_results, cong_stats):
    """Append imbalance findings to the existing summary.md."""
    summary_path = BASE_DIR / "summary.md"

    lines = [
        "",
        "---",
        "",
        "## H2H Capacity vs System Imbalance",
        "",
        f"**Overlap period**: {pd.to_datetime(df['date']).min().date()} to "
        f"{pd.to_datetime(df['date']).max().date()} "
        f"({df['date'].nunique()} days, {len(df)} hourly obs)",
        f"- Mean imbalance: {df['imbalance_mwh'].mean():+.1f} MWh/h",
        f"- Deficit hours: {df['is_deficit'].mean()*100:.1f}%",
        "",
        "### Correlations (Import Capacity vs Imbalance Volume)",
        "",
        "| Area | Pearson r | Spearman r |",
        "|------|-----------|------------|",
    ]

    for res in corr_results:
        lines.append(f"| {AREA_LABELS[res['area']]} | {res['pearson_r']:+.3f} | {res['spearman_r']:+.3f} |")

    valid = df[["total_import_cap", "imbalance_mwh"]].dropna()
    r, _ = stats.pearsonr(valid["total_import_cap"], valid["imbalance_mwh"])
    rs, _ = stats.spearmanr(valid["total_import_cap"], valid["imbalance_mwh"])
    lines.append(f"| **Total Import** | **{r:+.3f}** | **{rs:+.3f}** |")
    lines.append("")

    lines.append("### Deficit Rate by Congestion")
    lines.append("")
    lines.append("| Border | Deficit % Open | Deficit % Congested | |Imb| Open | |Imb| Congested |")
    lines.append("|--------|---------------|--------------------|-----------|--------------------|")
    for cs in cong_stats:
        lines.append(f"| {cs['area']} | {cs['deficit_open']:.1f}% | {cs['deficit_cong']:.1f}% | "
                     f"{cs['abs_open']:.1f} | {cs['abs_cong']:.1f} |")
    lines.append("")

    lines.append("### |Imbalance| by Number of Congested Borders")
    lines.append("")
    lines.append("| # Congested | Mean Imb | |Mean| | |Median| | |P90| | Deficit % | n |")
    lines.append("|-------------|----------|--------|---------|-------|-----------|---|")
    for count, grp in df.groupby("congested_in_count"):
        lines.append(f"| {int(count)} | {grp['imbalance_mwh'].mean():+.1f} | "
                     f"{grp['abs_imbalance'].mean():.1f} | {grp['abs_imbalance'].median():.1f} | "
                     f"{grp['abs_imbalance'].quantile(0.9):.1f} | "
                     f"{grp['is_deficit'].mean()*100:.1f}% | {len(grp)} |")
    lines.append("")

    lines.extend([
        "### Imbalance Plots",
        "",
        "6. `06_h2h_imbalance_correlation.png` - Correlation matrix: capacity vs imbalance",
        "7. `07_h2h_imbalance_scatter.png` - Per-area scatter: import cap vs imbalance",
        "8. `08_h2h_imbalance_congestion.png` - Box plots: imbalance by congestion state",
        "9. `09_h2h_imbalance_timeseries.png` - Weekly overlay: imbalance + capacity + congestion count",
        "10. `10_h2h_imbalance_deficit.png` - Deficit probability and magnitude analysis",
    ])

    with open(summary_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[+] Updated {summary_path}")


def main():
    print("=" * 60)
    print("H2H Capacity vs System Imbalance Analysis")
    print("=" * 60)

    df = load_and_merge()

    # Save merged dataset
    out_path = DATA_DIR / "h2h_imbalance_hourly.csv"
    df.to_csv(out_path, index=False)
    print(f"[+] Saved merged data: {out_path}")

    # Generate all plots
    plot_correlation_matrix(df)
    plot_capacity_vs_imbalance(df)
    plot_congestion_imbalance(df)
    plot_timeseries_overlay(df)
    plot_deficit_analysis(df)

    # Statistical analysis
    corr_results, cong_stats = compute_statistics(df)

    # Update summary
    update_summary(df, corr_results, cong_stats)

    print("\n--- Analysis complete ---")


if __name__ == "__main__":
    main()
