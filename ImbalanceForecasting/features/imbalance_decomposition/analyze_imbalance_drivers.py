"""
Imbalance Decomposition: Load Surprise + IDM Counter-Trading + H2H Filtering

Hypothesis: Imbalance = Load Surprise - IDM Counter-Trading (filtered to Slovakia-only)
- Load surprise creates a position (actual load != DA forecast)
- IDM counter-trading partially corrects it
- Only trades staying in Slovakia matter (H2H matrix filters cross-border)
- The uncorrected remainder = settlement imbalance

Approach A: Regression decomposition with incremental models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent.parent.parent
OUT = Path(__file__).resolve().parent
DATA_OUT = OUT / "data"
DATA_OUT.mkdir(exist_ok=True)

# =========================================================================
# 1. Load all data sources
# =========================================================================
print("[*] Loading data sources...")

# --- Load surprise (hourly) ---
load_df = pd.read_csv(BASE / "features" / "DamasLoad" / "load_data.csv", parse_dates=["datetime"])
load_df = load_df.rename(columns={"datetime": "timestamp_hour"})
load_df["timestamp_hour"] = pd.to_datetime(load_df["timestamp_hour"])
# forecast_error_mw = actual - forecast
# Positive = more demand than expected -> system should be SHORT
print(f"  [+] Load data: {len(load_df)} rows, {load_df['timestamp_hour'].min()} to {load_df['timestamp_hour'].max()}")

# --- IDM 60-min trades ---
idm_df = pd.read_csv(BASE / "data" / "clean" / "market" / "intraday" / "idm_60min.csv", sep=";")
# Parse numeric columns (have commas as thousands separators)
for col in ["Buy Orders (MWh)", "Sell Orders (MWh)", "Buy trades (MWh)",
            "Sell trades (MWh)", "Traded Quantity Difference (MWh)",
            "Weighted average price of all trades (EUR/MWh)", "Total Traded Quantity (MWh)"]:
    if col in idm_df.columns:
        idm_df[col] = idm_df[col].astype(str).str.replace(",", "").astype(float)

# Build timestamp
idm_df["timestamp_hour"] = pd.to_datetime(idm_df["Delivery day"]) + pd.to_timedelta(idm_df["Period number"] - 1, unit="h")
idm_df = idm_df.rename(columns={
    "Buy trades (MWh)": "idm_buy_mwh",
    "Sell trades (MWh)": "idm_sell_mwh",
    "Traded Quantity Difference (MWh)": "idm_net_mwh",
    "Total Traded Quantity (MWh)": "idm_total_mwh",
    "Weighted average price of all trades (EUR/MWh)": "idm_vwap",
    "Buy Orders (MWh)": "idm_buy_orders_mwh",
    "Sell Orders (MWh)": "idm_sell_orders_mwh",
})
idm_cols = ["timestamp_hour", "idm_buy_mwh", "idm_sell_mwh", "idm_net_mwh",
            "idm_total_mwh", "idm_vwap", "idm_buy_orders_mwh", "idm_sell_orders_mwh"]
idm_df = idm_df[idm_cols].copy()
print(f"  [+] IDM data: {len(idm_df)} rows, {idm_df['timestamp_hour'].min()} to {idm_df['timestamp_hour'].max()}")

# --- Hourly market prices (has imbalance aggregated to hourly) ---
mkt_df = pd.read_csv(BASE / "MarketPriceGap" / "data" / "processed" / "hourly_market_prices.csv",
                      parse_dates=["timestamp_hour"])
# imbalance_mwh = sum of 4 quarter-hour imbalances
mkt_cols = ["timestamp_hour", "imbalance_mwh", "da_price", "imb_settlement_price"]
mkt_df = mkt_df[mkt_cols].copy()
print(f"  [+] Market data: {len(mkt_df)} rows, {mkt_df['timestamp_hour'].min()} to {mkt_df['timestamp_hour'].max()}")

# --- H2H + Imbalance (already merged) ---
h2h_path = BASE / "MarketPriceGap" / "features" / "h2h_capacity" / "data" / "h2h_imbalance_hourly.csv"
h2h_df = pd.read_csv(h2h_path)
h2h_df["timestamp_hour"] = pd.to_datetime(h2h_df["tradeday"]) + pd.to_timedelta(h2h_df["hour"], unit="h")
h2h_keep = ["timestamp_hour", "total_import_cap", "total_export_cap", "net_capacity",
            "congested_in_count",
            "ceps_in", "ceps_out", "apg_in", "apg_out",
            "mavir_in", "mavir_out", "pse_in", "pse_out", "de_in", "de_out",
            "ceps_congested_in", "apg_congested_in", "mavir_congested_in",
            "pse_congested_in", "de_congested_in"]
h2h_df = h2h_df[[c for c in h2h_keep if c in h2h_df.columns]].copy()
print(f"  [+] H2H data: {len(h2h_df)} rows, {h2h_df['timestamp_hour'].min()} to {h2h_df['timestamp_hour'].max()}")

# =========================================================================
# 2. Merge datasets
# =========================================================================
print("\n[*] Merging datasets...")

# Core merge: load + imbalance
df = load_df[["timestamp_hour", "forecast_error_mw", "actual_load_mw", "forecast_load_mw", "hour"]].merge(
    mkt_df, on="timestamp_hour", how="inner"
)
print(f"  [+] Load + Imbalance: {len(df)} rows")

# Add IDM
df_idm = df.merge(idm_df, on="timestamp_hour", how="inner")
print(f"  [+] + IDM: {len(df_idm)} rows ({df_idm['timestamp_hour'].min()} to {df_idm['timestamp_hour'].max()})")

# Add H2H (limited coverage)
df_full = df_idm.merge(h2h_df, on="timestamp_hour", how="inner")
print(f"  [+] + H2H: {len(df_full)} rows ({df_full['timestamp_hour'].min()} to {df_full['timestamp_hour'].max()})")

# =========================================================================
# 3. Feature engineering
# =========================================================================
print("\n[*] Engineering features...")

# Load surprise (already have forecast_error_mw = actual - forecast)
# Rename for clarity
df_idm["load_surprise_mw"] = df_idm["forecast_error_mw"]
df_full["load_surprise_mw"] = df_full["forecast_error_mw"]

# IDM order imbalance (buy orders - sell orders as a proxy for demand pressure)
df_idm["idm_order_imbalance_mwh"] = df_idm["idm_buy_orders_mwh"] - df_idm["idm_sell_orders_mwh"]
df_full["idm_order_imbalance_mwh"] = df_full["idm_buy_orders_mwh"] - df_full["idm_sell_orders_mwh"]

# "Uncorrected surprise" = load surprise that IDM didn't fix
# If load_surprise > 0 (more demand), IDM net buy should be positive (buying to cover)
# The residual that wasn't covered goes to imbalance
# We scale load surprise from MW to MWh (1 hour -> factor 1)
df_idm["uncorrected_surprise"] = df_idm["load_surprise_mw"] - df_idm["idm_net_mwh"]
df_full["uncorrected_surprise"] = df_full["load_surprise_mw"] - df_full["idm_net_mwh"]

# H2H features for full model
if "net_capacity" in df_full.columns:
    # Interaction: load surprise * congestion (less correction possible when congested)
    df_full["surprise_x_congestion"] = df_full["load_surprise_mw"] * df_full["congested_in_count"]
    # Normalized capacity available
    df_full["net_cap_normalized"] = df_full["net_capacity"] / df_full["net_capacity"].std()

# =========================================================================
# 4. Incremental regression models
# =========================================================================
print("\n[*] Running incremental regressions...")
print("=" * 80)

results = []

def run_model(name, data, features, target="imbalance_mwh"):
    """Run OLS regression and report results."""
    mask = data[features + [target]].dropna().index
    X = data.loc[mask, features].values
    y = data.loc[mask, target].values
    n = len(y)

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    r = np.sign(np.corrcoef(y, y_pred)[0, 1]) * np.sqrt(abs(r2))
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))

    print(f"\n--- Model: {name} ---")
    print(f"  N = {n}, R2 = {r2:.4f}, r = {r:.4f}, MAE = {mae:.2f} MWh, RMSE = {rmse:.2f} MWh")
    print(f"  Features:")
    for feat, coef in zip(features, model.coef_):
        print(f"    {feat:35s} coef = {coef:+.6f}")
    print(f"    {'intercept':35s} = {model.intercept_:+.4f}")

    results.append({
        "model": name, "n_obs": n, "n_features": len(features),
        "r2": r2, "r": r, "mae": mae, "rmse": rmse,
        "intercept": model.intercept_,
    })
    for feat, coef in zip(features, model.coef_):
        results[-1][f"coef_{feat}"] = coef

    return model, y, y_pred, mask


# --- Model 1: Load surprise only (baseline, full 2025 period) ---
m1, y1, yp1, mask1 = run_model(
    "M1: Load Surprise Only",
    df_idm,
    ["load_surprise_mw"]
)

# --- Model 2: Load surprise + IDM net volume ---
m2, y2, yp2, mask2 = run_model(
    "M2: + IDM Net Volume",
    df_idm,
    ["load_surprise_mw", "idm_net_mwh"]
)

# --- Model 3: Load surprise + IDM details ---
m3, y3, yp3, mask3 = run_model(
    "M3: + IDM Buy/Sell Split",
    df_idm,
    ["load_surprise_mw", "idm_buy_mwh", "idm_sell_mwh"]
)

# --- Model 4: Uncorrected surprise (single composite feature) ---
m4, y4, yp4, mask4 = run_model(
    "M4: Uncorrected Surprise",
    df_idm,
    ["uncorrected_surprise"]
)

# --- Model 5: Kitchen sink IDM ---
m5, y5, yp5, mask5 = run_model(
    "M5: Load + IDM + Orders",
    df_idm,
    ["load_surprise_mw", "idm_net_mwh", "idm_total_mwh", "idm_order_imbalance_mwh"]
)

# --- Models with H2H (limited period) ---
if len(df_full) > 100:
    print("\n" + "=" * 80)
    print("[*] H2H-augmented models (limited period)...")

    # Rerun M1 on H2H period for fair comparison
    m1h, y1h, yp1h, mask1h = run_model(
        "M1h: Load Surprise (H2H period)",
        df_full,
        ["load_surprise_mw"]
    )

    # M2h: load + IDM on H2H period
    m2h, y2h, yp2h, mask2h = run_model(
        "M2h: Load + IDM (H2H period)",
        df_full,
        ["load_surprise_mw", "idm_net_mwh"]
    )

    # M6: Load + IDM + H2H capacity
    m6, y6, yp6, mask6 = run_model(
        "M6: + H2H Net Capacity",
        df_full,
        ["load_surprise_mw", "idm_net_mwh", "net_capacity"]
    )

    # M7: Load + IDM + H2H + congestion interaction
    m7, y7, yp7, mask7 = run_model(
        "M7: + Congestion Interaction",
        df_full,
        ["load_surprise_mw", "idm_net_mwh", "net_capacity",
         "congested_in_count", "surprise_x_congestion"]
    )

    # M8: Full model with border-level detail
    m8, y8, yp8, mask8 = run_model(
        "M8: Full (borders + IDM)",
        df_full,
        ["load_surprise_mw", "idm_net_mwh",
         "ceps_in", "ceps_out", "apg_in", "apg_out",
         "mavir_in", "mavir_out", "net_capacity", "congested_in_count"]
    )

# =========================================================================
# 5. Diagnostic: correlation matrix
# =========================================================================
print("\n" + "=" * 80)
print("[*] Key correlations (2025 IDM period)...")

corr_cols = ["imbalance_mwh", "load_surprise_mw", "idm_net_mwh",
             "idm_total_mwh", "idm_order_imbalance_mwh", "uncorrected_surprise"]
corr_matrix = df_idm[corr_cols].corr()
print("\nCorrelation with imbalance_mwh:")
for col in corr_cols[1:]:
    print(f"  {col:35s} r = {corr_matrix.loc['imbalance_mwh', col]:+.4f}")

# =========================================================================
# 6. Time-of-day breakdown
# =========================================================================
print("\n[*] R-squared by hour of day...")
hourly_r2 = []
for h in range(24):
    subset = df_idm[df_idm["hour"] == h].dropna(subset=["load_surprise_mw", "idm_net_mwh", "imbalance_mwh"])
    if len(subset) < 30:
        continue
    X = subset[["load_surprise_mw", "idm_net_mwh"]].values
    y = subset["imbalance_mwh"].values
    m = LinearRegression().fit(X, y)
    r2 = r2_score(y, m.predict(X))

    # Also just load surprise alone
    X1 = subset[["load_surprise_mw"]].values
    m1_h = LinearRegression().fit(X1, y)
    r2_load = r2_score(y, m1_h.predict(X1))

    hourly_r2.append({"hour": h, "r2_load_only": r2_load, "r2_load_idm": r2,
                       "r2_gain": r2 - r2_load, "n": len(subset)})
    print(f"  Hour {h:2d}: R2_load={r2_load:.3f}  R2_load+IDM={r2:.3f}  gain={r2 - r2_load:+.3f}  (n={len(subset)})")

hourly_r2_df = pd.DataFrame(hourly_r2)

# =========================================================================
# 7. Save results
# =========================================================================
print("\n[*] Saving results...")

results_df = pd.DataFrame(results)
results_df.to_csv(DATA_OUT / "regression_results.csv", index=False)
print(f"  [+] {DATA_OUT / 'regression_results.csv'}")

hourly_r2_df.to_csv(DATA_OUT / "hourly_r2_breakdown.csv", index=False)
print(f"  [+] {DATA_OUT / 'hourly_r2_breakdown.csv'}")

corr_matrix.to_csv(DATA_OUT / "correlation_matrix.csv")
print(f"  [+] {DATA_OUT / 'correlation_matrix.csv'}")

# =========================================================================
# 8. Visualization
# =========================================================================
print("\n[*] Creating plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Imbalance Decomposition: Load Surprise + IDM Counter-Trading + H2H",
             fontsize=14, fontweight="bold")

# --- Panel 1: Model comparison bar chart ---
ax = axes[0, 0]
model_names = results_df["model"].str.replace(r"M\d+h?: ", "", regex=True)
colors = ["#4477AA" if "H2H" not in n and "border" not in n.lower() and "Congestion" not in n
          else "#AA3377" for n in results_df["model"]]
bars = ax.barh(range(len(results_df)), results_df["r2"], color=colors, edgecolor="black", linewidth=0.5)
ax.set_yticks(range(len(results_df)))
ax.set_yticklabels(model_names, fontsize=8)
ax.set_xlabel("R-squared")
ax.set_title("Model Comparison (R2)")
for i, (r2, n) in enumerate(zip(results_df["r2"], results_df["n_obs"])):
    ax.text(r2 + 0.002, i, f"{r2:.3f} (n={n})", va="center", fontsize=7)
ax.invert_yaxis()

# --- Panel 2: Load surprise vs imbalance scatter ---
ax = axes[0, 1]
sample = df_idm.sample(min(3000, len(df_idm)), random_state=42)
ax.scatter(sample["load_surprise_mw"], sample["imbalance_mwh"], alpha=0.15, s=8, c="#4477AA")
ax.axhline(0, color="black", linewidth=0.5)
ax.axvline(0, color="black", linewidth=0.5)
r_val = df_idm[["load_surprise_mw", "imbalance_mwh"]].corr().iloc[0, 1]
ax.set_xlabel("Load Surprise (MW)")
ax.set_ylabel("Imbalance (MWh)")
ax.set_title(f"Load Surprise vs Imbalance (r={r_val:.3f})")

# --- Panel 3: IDM net volume vs imbalance ---
ax = axes[0, 2]
ax.scatter(sample["idm_net_mwh"], sample["imbalance_mwh"], alpha=0.15, s=8, c="#EE6677")
ax.axhline(0, color="black", linewidth=0.5)
ax.axvline(0, color="black", linewidth=0.5)
r_val2 = df_idm[["idm_net_mwh", "imbalance_mwh"]].corr().iloc[0, 1]
ax.set_xlabel("IDM Net Volume (MWh) [Buy - Sell]")
ax.set_ylabel("Imbalance (MWh)")
ax.set_title(f"IDM Net Volume vs Imbalance (r={r_val2:.3f})")

# --- Panel 4: Uncorrected surprise vs imbalance ---
ax = axes[1, 0]
ax.scatter(sample["uncorrected_surprise"], sample["imbalance_mwh"], alpha=0.15, s=8, c="#228833")
ax.axhline(0, color="black", linewidth=0.5)
ax.axvline(0, color="black", linewidth=0.5)
r_val3 = df_idm[["uncorrected_surprise", "imbalance_mwh"]].corr().iloc[0, 1]
ax.set_xlabel("Uncorrected Surprise (MW - MWh)")
ax.set_ylabel("Imbalance (MWh)")
ax.set_title(f"Uncorrected Surprise vs Imbalance (r={r_val3:.3f})")

# --- Panel 5: R2 by hour of day ---
ax = axes[1, 1]
ax.bar(hourly_r2_df["hour"] - 0.2, hourly_r2_df["r2_load_only"], width=0.4,
       label="Load only", color="#4477AA", edgecolor="black", linewidth=0.5)
ax.bar(hourly_r2_df["hour"] + 0.2, hourly_r2_df["r2_load_idm"], width=0.4,
       label="Load + IDM", color="#EE6677", edgecolor="black", linewidth=0.5)
ax.set_xlabel("Hour of Day")
ax.set_ylabel("R-squared")
ax.set_title("Explanatory Power by Hour")
ax.legend(fontsize=8)
ax.set_xticks(range(0, 24, 3))

# --- Panel 6: IDM volume distribution by load surprise sign ---
ax = axes[1, 2]
pos_surprise = df_idm[df_idm["load_surprise_mw"] > 25]["idm_net_mwh"]
neg_surprise = df_idm[df_idm["load_surprise_mw"] < -25]["idm_net_mwh"]
bins = np.linspace(-200, 200, 50)
ax.hist(pos_surprise, bins=bins, alpha=0.5, color="#EE6677", label=f"Load surprise > +25 MW (n={len(pos_surprise)})")
ax.hist(neg_surprise, bins=bins, alpha=0.5, color="#4477AA", label=f"Load surprise < -25 MW (n={len(neg_surprise)})")
ax.axvline(pos_surprise.mean(), color="#EE6677", linestyle="--", linewidth=2)
ax.axvline(neg_surprise.mean(), color="#4477AA", linestyle="--", linewidth=2)
ax.set_xlabel("IDM Net Volume (MWh)")
ax.set_ylabel("Count")
ax.set_title("IDM Response to Load Surprise")
ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(OUT / "01_imbalance_decomposition.png", dpi=150, bbox_inches="tight")
print(f"  [+] {OUT / '01_imbalance_decomposition.png'}")

# =========================================================================
# 9. Print summary
# =========================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nHypothesis: Imbalance = Load Surprise - IDM Counter-Trading (H2H filtered)")
print(f"\nKey findings:")

m1_r2 = results_df[results_df["model"].str.contains("M1:")]["r2"].values[0]
m2_r2 = results_df[results_df["model"].str.contains("M2:")]["r2"].values[0]
m4_r2 = results_df[results_df["model"].str.contains("M4:")]["r2"].values[0]
m5_r2 = results_df[results_df["model"].str.contains("M5:")]["r2"].values[0]

print(f"  1. Load surprise alone:       R2 = {m1_r2:.4f}")
print(f"  2. + IDM net volume:           R2 = {m2_r2:.4f} (gain: {m2_r2 - m1_r2:+.4f})")
print(f"  3. Uncorrected surprise:       R2 = {m4_r2:.4f}")
print(f"  4. Full IDM model:             R2 = {m5_r2:.4f} (gain: {m5_r2 - m1_r2:+.4f})")

if len(df_full) > 100:
    m1h_r2 = results_df[results_df["model"].str.contains("M1h:")]["r2"].values[0]
    m7_rows = results_df[results_df["model"].str.contains("M7:")]
    m8_rows = results_df[results_df["model"].str.contains("M8:")]
    if len(m7_rows) > 0:
        m7_r2 = m7_rows["r2"].values[0]
        print(f"  5. + H2H + congestion:         R2 = {m7_r2:.4f} (gain vs load: {m7_r2 - m1h_r2:+.4f})")
    if len(m8_rows) > 0:
        m8_r2 = m8_rows["r2"].values[0]
        print(f"  6. Full (borders + IDM):        R2 = {m8_r2:.4f} (gain vs load: {m8_r2 - m1h_r2:+.4f})")

# IDM counter-trading direction check
corr_load_idm = df_idm[["load_surprise_mw", "idm_net_mwh"]].corr().iloc[0, 1]
print(f"\n  Load surprise <-> IDM net: r = {corr_load_idm:+.4f}")
if corr_load_idm > 0:
    print("  -> IDM net buying increases with load surprise (expected: counter-trading)")
else:
    print("  -> IDM net buying DECREASES with load surprise (unexpected direction!)")

mean_pos = df_idm[df_idm["load_surprise_mw"] > 25]["idm_net_mwh"].mean()
mean_neg = df_idm[df_idm["load_surprise_mw"] < -25]["idm_net_mwh"].mean()
print(f"  When load surprise > +25 MW: mean IDM net = {mean_pos:+.1f} MWh")
print(f"  When load surprise < -25 MW: mean IDM net = {mean_neg:+.1f} MWh")

print("\n[+] Analysis complete.")
