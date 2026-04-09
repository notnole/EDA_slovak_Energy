"""
02_feature_engineering.py - Compute daily feature matrix for DA decision matrix.

STRICT CAUSALITY: only features available before DA auction gate closure (~10:00 D-1).

  DA-AVAILABLE (features):
    - Forecasts for D+1: load shape, solar shape, temperature
    - Yesterday's REALIZED: prices, load, production mix, flows, volatility
    - 7-day rolling averages (lagged)
    - Calendar: day_of_week, is_weekend, month

  NOT DA-AVAILABLE (targets / realized only):
    - Same-day DA price, IDM VWAP, spread (= targets for backtesting)
    - Same-day production mix, flows, actual load (= needed only to compute lags)

Input:  da_master_hourly.csv
Output: da_daily_features.csv (one row per day, only causal features + targets)
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "MarketPriceGap" / "da_decision_matrix" / "data"

MASTER_FILE = DATA_DIR / "da_master_hourly.csv"


def load_master():
    print("[*] Loading master hourly dataset...")
    df = pd.read_csv(MASTER_FILE, parse_dates=[0], index_col=0)
    print(f"[+]   {len(df)} rows, {df.index.min()} to {df.index.max()}")
    return df


def compute_daily_features(df):
    """Compute day-level features and targets."""
    print("[*] Computing daily features and targets...")

    # Ensure date column
    df["date_col"] = df.index.date

    # Group by date
    daily_groups = df.groupby("date_col")
    dates = sorted(df["date_col"].unique())

    rows = []
    for d in dates:
        day_data = daily_groups.get_group(d)
        row = {"date": pd.Timestamp(d)}

        # Skip days with too few hours
        if len(day_data) < 20:
            continue

        # ==========================================================
        # A. LOAD-BASED FEATURES (from DAMAS forecast for this day)
        # ==========================================================
        lf = day_data["load_forecast_mw"]
        la = day_data["load_actual_mw"]
        if lf.notna().sum() >= 20:
            row["load_fcst_mean"] = lf.mean()
            row["load_fcst_peak"] = lf.max()
            row["load_fcst_peak_hour"] = lf.idxmax().hour if lf.notna().any() else np.nan
            row["load_fcst_offpeak"] = lf.min()
            row["load_fcst_range"] = lf.max() - lf.min()
        if la.notna().sum() >= 20:
            row["load_actual_mean"] = la.mean()

        # ==========================================================
        # B. SOLAR-BASED FEATURES (from TSO forecast)
        # ==========================================================
        sf = day_data["sk_f_solar"]
        sa = day_data["sk_a_solar"]
        if sf.notna().sum() >= 5:
            row["solar_fcst_total"] = sf.sum()  # daily MWh proxy (hourly MW * 1h)
            row["solar_fcst_peak"] = sf.max()
            row["solar_fcst_peak_hour"] = sf.idxmax().hour if sf.notna().any() else np.nan
            # Max hour-to-hour ramp (cloud risk proxy)
            sf_diff = sf.diff().abs()
            row["solar_fcst_max_ramp"] = sf_diff.max()
        if sa.notna().sum() >= 5:
            row["solar_actual_total"] = sa.sum()

        # Solar share of load
        if "load_fcst_mean" in row and row.get("solar_fcst_total", 0) > 0 and row["load_fcst_mean"] > 0:
            row["solar_share_of_load"] = (row["solar_fcst_total"] / 24) / row["load_fcst_mean"]
        else:
            row["solar_share_of_load"] = 0.0

        # ==========================================================
        # C. TEMPERATURE-BASED FEATURES
        # ==========================================================
        tf = day_data["temp_forecast"]
        ta = day_data["temp_actual"]
        if tf.notna().sum() >= 10:
            row["temp_fcst_mean"] = tf.mean()
            row["temp_fcst_min"] = tf.min()
            row["temp_fcst_max"] = tf.max()
            row["temp_fcst_range"] = tf.max() - tf.min()
            row["cold_spell"] = 1 if tf.min() < 0 else 0
        if ta.notna().sum() >= 10:
            row["temp_actual_mean"] = ta.mean()

        # ==========================================================
        # D. SUPPLY-SIDE FEATURES (production mix for this day)
        # ==========================================================
        for col, name in [
            ("sk_a_nuclear", "nuclear_mw"),
            ("sk_a_hydroreservoir", "hydro_reservoir_mw"),
            ("sk_a_hydrorunriver", "hydro_runriver_mw"),
            ("sk_a_hydropump", "hydro_pump_mw"),
            ("sk_a_natgas", "gas_mw"),
            ("sk_a_lignite", "lignite_mw"),
        ]:
            series = day_data.get(col)
            if series is not None and series.notna().sum() >= 10:
                row[name + "_mean"] = series.mean()

        # Total hydro
        hydro_cols = ["sk_a_hydroreservoir", "sk_a_hydrorunriver", "sk_a_hydropump"]
        hydro_sum = day_data[hydro_cols].sum(axis=1)
        if hydro_sum.notna().sum() >= 10:
            row["hydro_total_mw_mean"] = hydro_sum.mean()

        # Net import
        ni = day_data.get("net_import")
        if ni is not None and ni.notna().sum() >= 10:
            row["net_import_mean"] = ni.mean()

        # Cross-border flows
        for flow_col in ["net_flow_cz", "net_flow_pl", "net_flow_hu"]:
            fl = day_data.get(flow_col)
            if fl is not None and fl.notna().sum() >= 10:
                row[flow_col + "_mean"] = fl.mean()

        # ==========================================================
        # E. DA PRICE TARGETS (realized - for backtesting only)
        # ==========================================================
        da = day_data["da_price"]
        if da.notna().sum() >= 20:
            row["da_price_mean"] = da.mean()
            row["da_price_max"] = da.max()
            row["da_price_max_hour"] = da.idxmax().hour
            row["da_price_min"] = da.min()
            row["da_price_min_hour"] = da.idxmin().hour
            row["da_price_range"] = da.max() - da.min()
            row["da_price_std"] = da.std()
            row["da_price_median"] = da.median()
            # Peak timing classification
            row["peak_in_morning"] = 1 if da.idxmax().hour < 12 else 0
            row["peak_in_evening"] = 1 if da.idxmax().hour >= 16 else 0
            # Extreme price flag
            row["extreme_price_flag"] = 1 if (da.max() > 200 or da.min() < 10) else 0

        # ==========================================================
        # F. IDM PRICE TARGETS
        # ==========================================================
        idm = day_data.get("idm_vwap")
        if idm is not None and idm.notna().sum() >= 10:
            row["idm_vwap_mean"] = idm.mean()

        # DA-IDM spread
        spread = day_data.get("spread_da_idm")
        if spread is not None and spread.notna().sum() >= 10:
            row["spread_da_idm_mean"] = spread.mean()
            row["spread_positive"] = 1 if spread.mean() > 0 else 0

        # ==========================================================
        # G. CALENDAR
        # ==========================================================
        row["day_of_week"] = pd.Timestamp(d).dayofweek
        row["is_weekend"] = 1 if pd.Timestamp(d).dayofweek >= 5 else 0
        row["month"] = pd.Timestamp(d).month

        rows.append(row)

    daily = pd.DataFrame(rows)
    daily = daily.set_index("date").sort_index()
    print(f"[+]   {len(daily)} days computed")
    return daily


def add_lagged_features(daily):
    """Add lagged (yesterday, 7-day rolling) features -- all causal."""
    print("[*] Adding lagged features (yesterday, 7-day rolling)...")

    # --- Yesterday's realized values ---
    lag_cols = {
        "da_price_mean": "da_price_yesterday",
        "da_price_std": "da_volatility_yesterday",
        "da_price_range": "da_range_yesterday",
        "da_price_max": "da_max_yesterday",
        "idm_vwap_mean": "idm_vwap_yesterday",
        "spread_da_idm_mean": "spread_yesterday",
        "load_actual_mean": "load_actual_yesterday",
        "nuclear_mw_mean": "nuclear_yesterday",
        "hydro_total_mw_mean": "hydro_yesterday",
        "gas_mw_mean": "gas_yesterday",
        "solar_actual_total": "solar_actual_yesterday",
        "net_import_mean": "net_import_yesterday",
    }
    for src, dst in lag_cols.items():
        if src in daily.columns:
            daily[dst] = daily[src].shift(1)

    # --- 7-day rolling means ---
    roll_cols = {
        "da_price_mean": "da_price_7d",
        "da_price_std": "da_volatility_7d",
        "da_price_range": "da_range_7d",
        "idm_vwap_mean": "idm_vwap_7d",
        "spread_da_idm_mean": "spread_7d",
        "load_actual_mean": "load_actual_7d",
    }
    for src, dst in roll_cols.items():
        if src in daily.columns:
            daily[dst] = daily[src].rolling(7, min_periods=3).mean().shift(1)

    # --- Deviation from normal ---
    # Load forecast vs 7-day actual load average
    if "load_fcst_mean" in daily.columns and "load_actual_7d" in daily.columns:
        daily["load_fcst_vs_7d"] = daily["load_fcst_mean"] - daily["load_actual_7d"]

    # Temperature forecast vs 7-day actual temp average
    if "temp_fcst_mean" in daily.columns and "temp_actual_mean" in daily.columns:
        daily["temp_7d_mean"] = daily["temp_actual_mean"].rolling(7, min_periods=3).mean().shift(1)
        daily["temp_fcst_vs_7d"] = daily["temp_fcst_mean"] - daily["temp_7d_mean"]

    # Solar forecast vs 7-day actual solar average
    if "solar_fcst_total" in daily.columns and "solar_actual_total" in daily.columns:
        daily["solar_7d_mean"] = daily["solar_actual_total"].rolling(7, min_periods=3).mean().shift(1)
        daily["solar_fcst_vs_7d"] = daily["solar_fcst_total"] - daily["solar_7d_mean"]

    # --- Same day-of-week features (last week same DoW) ---
    if "load_fcst_mean" in daily.columns:
        daily["load_fcst_vs_last_week"] = daily["load_fcst_mean"] - daily["load_fcst_mean"].shift(7)

    # --- Volatility targets ---
    if "da_price_std" in daily.columns:
        q75 = daily["da_price_std"].quantile(0.75)
        daily["high_volatility"] = (daily["da_price_std"] > q75).astype(int)

    # --- DROP same-day realized columns (NOT available at DA decision time) ---
    # These were only needed to compute yesterday lags and 7d rolling averages.
    same_day_realized = [
        "load_actual_mean",
        "solar_actual_total",
        "temp_actual_mean",
        "nuclear_mw_mean", "hydro_reservoir_mw_mean", "hydro_runriver_mw_mean",
        "hydro_pump_mw_mean", "gas_mw_mean", "lignite_mw_mean", "hydro_total_mw_mean",
        "net_import_mean",
        "net_flow_cz_mean", "net_flow_pl_mean", "net_flow_hu_mean",
        "temp_7d_mean", "solar_7d_mean",  # intermediate calc columns
    ]
    dropped = [c for c in same_day_realized if c in daily.columns]
    daily = daily.drop(columns=dropped)
    print(f"[*] Dropped {len(dropped)} same-day realized columns (not DA-available):")
    for c in dropped:
        print(f"      - {c}")

    # Classify remaining columns
    target_cols = [
        "da_price_mean", "da_price_max", "da_price_max_hour", "da_price_min",
        "da_price_min_hour", "da_price_range", "da_price_std", "da_price_median",
        "peak_in_morning", "peak_in_evening", "extreme_price_flag",
        "idm_vwap_mean", "spread_da_idm_mean", "spread_positive", "high_volatility",
    ]
    features = [c for c in daily.columns if c not in target_cols]
    targets = [c for c in daily.columns if c in target_cols]
    print(f"\n[+]   {len(features)} DA-available features + {len(targets)} targets = {len(daily.columns)} total")
    print(f"[*]   Features: {features}")
    return daily


def main():
    print("=" * 70)
    print("DA Decision Matrix - Step 2: Feature Engineering")
    print("=" * 70)

    df = load_master()
    daily = compute_daily_features(df)
    daily = add_lagged_features(daily)

    # Filter to overlap period (where we have all key signals)
    key_cols = ["da_price_mean", "load_fcst_mean", "temp_fcst_mean"]
    mask = daily[key_cols].notna().all(axis=1)
    daily_filtered = daily[mask].copy()

    print(f"\n[+] Final dataset: {len(daily_filtered)} days with all key signals")
    print(f"[*] Period: {daily_filtered.index.min()} to {daily_filtered.index.max()}")
    print(f"[*] Columns ({len(daily_filtered.columns)}):")
    for col in daily_filtered.columns:
        n_valid = daily_filtered[col].notna().sum()
        print(f"    {col:35s}: {n_valid:4d} valid ({100*n_valid/len(daily_filtered):.0f}%)")

    # Save both full and filtered
    daily.to_csv(DATA_DIR / "da_daily_features_full.csv")
    daily_filtered.to_csv(DATA_DIR / "da_daily_features.csv")
    print(f"\n[+] Saved: {DATA_DIR / 'da_daily_features.csv'} ({len(daily_filtered)} rows)")
    print(f"[+] Saved: {DATA_DIR / 'da_daily_features_full.csv'} ({len(daily)} rows)")

    # Quick stats on key targets
    print("\n--- Target Statistics ---")
    target_cols = ["da_price_mean", "da_price_std", "da_price_range",
                   "spread_da_idm_mean", "spread_positive",
                   "da_price_max_hour", "peak_in_evening", "high_volatility",
                   "extreme_price_flag"]
    for col in target_cols:
        if col in daily_filtered.columns:
            s = daily_filtered[col]
            if s.nunique() <= 2:
                print(f"  {col:30s}: {s.mean():.1%} positive rate")
            else:
                print(f"  {col:30s}: mean={s.mean():.1f}, std={s.std():.1f}, "
                      f"min={s.min():.1f}, max={s.max():.1f}")

    return daily_filtered


if __name__ == "__main__":
    main()
