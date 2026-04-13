"""
Verify production predict.py matches training pipeline exactly.
================================================================

Loads CSVs the same way train_multi_lead.py does, feeds them through
both the training pipeline and production predict.py, and asserts
that every feature value matches to floating-point precision.

Run: python ImbalanceForcastingProd/scripts/production/verify_predict.py
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parents[1]  # ImbalanceForcastingProd/
REPO_ROOT = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(BASE_DIR / "scripts" / "training"))

from predict import SpreadPredictor, SELECTED_FEATURES, SCADA_SHIFT
from train_multi_lead import load_all_data, build_features
import train_multi_lead as tml


def load_raw_dataframes():
    """Load raw DataFrames the same way the production caller would.

    This is the ONLY place CSVs are read — predict.py never touches files.
    """

    # Regulation 3-min
    reg = pd.read_csv(REPO_ROOT / "data" / "features" / "regulation_3min.csv",
                      parse_dates=['datetime']).set_index('datetime').sort_index()

    # Load 3-min
    load = pd.read_csv(REPO_ROOT / "data" / "features" / "load_3min.csv",
                       parse_dates=['datetime']).set_index('datetime').sort_index()

    # Production 3-min
    prod_path = REPO_ROOT / "data" / "features" / "production_3min.csv"
    prod = pd.read_csv(prod_path, parse_dates=['datetime']).set_index('datetime').sort_index() if prod_path.exists() else None

    # Export/Import 3-min
    xi_path = REPO_ROOT / "data" / "features" / "export_import_3min.csv"
    xi = pd.read_csv(xi_path, parse_dates=['datetime']).set_index('datetime').sort_index() if xi_path.exists() else None

    # Solar hourly
    solar = pd.read_csv(REPO_ROOT / "data" / "clean" / "solar" / "solar_hourly.csv",
                        parse_dates=['datetime']).set_index('datetime').sort_index()

    # DAMAS load
    damas = pd.read_csv(REPO_ROOT / "features" / "DamasLoad" / "load_data.csv",
                        parse_dates=['datetime']).set_index('datetime').sort_index()

    # DA prices
    da = pd.read_csv(REPO_ROOT / "features" / "DamasPrices" / "data" / "da_prices.csv",
                     parse_dates=['datetime']).set_index('datetime').sort_index()

    # Market spreads
    mkt = pd.read_csv(REPO_ROOT / "MarketPriceGap" / "data" / "processed" / "hourly_market_prices.csv",
                      parse_dates=['timestamp_hour']).set_index('timestamp_hour').sort_index()

    # Weather (multi-city + DA forecast)
    mc = pd.read_csv(REPO_ROOT / "data" / "Bardejov" / "Weather" / "slovakia_multi_city_weather.csv",
                     parse_dates=['time']).set_index('time').sort_index()

    weather = mc[['temp_national_mean', 'temp_national_spread', 'temp_bratislava',
                   'wind_national_mean', 'radiation_national_mean',
                   'pressure_national_mean', 'temp_bardejov', 'cloud_bardejov']].copy()
    weather = weather.rename(columns={'temp_bardejov': 'temperature', 'cloud_bardejov': 'cloudcover'})

    da_wx_path = REPO_ROOT / "data" / "Bardejov" / "Weather" / "bardejov_da_forecasts.csv"
    if da_wx_path.exists():
        da_wx = pd.read_csv(da_wx_path, parse_dates=['time']).set_index('time').sort_index()
        da_wx = da_wx[~da_wx.index.duplicated(keep='last')]
        for col in ['gfs_seamless_temp_da1', 'best_match_temp_da1']:
            if col in da_wx.columns:
                weather = weather.join(da_wx[[col]], how='left')
                weather = weather.rename(columns={col: 'temp_forecast_da'})
                break

    # Load nowcast OOS — dedup each horizon BEFORE joining (matches load_all_data)
    oos_dir = REPO_ROOT / "LoadAnalysis" / "nowcast_5h" / "tuning" / "oos_predictions"
    nc = pd.read_csv(oos_dir / "h2_oos_predictions.csv",
                     parse_dates=['datetime']).set_index('datetime').sort_index()
    nc = nc[~nc.index.duplicated(keep='last')]
    nowcast = nc[['predicted_error', 'actual_error']].copy()
    nowcast.columns = ['nowcast_pred_error', 'nowcast_actual_error']

    for h in [3, 4, 5]:
        h_path = oos_dir / f"h{h}_oos_predictions.csv"
        if h_path.exists():
            nc_h = pd.read_csv(h_path, parse_dates=['datetime']).set_index('datetime').sort_index()
            nc_h = nc_h[~nc_h.index.duplicated(keep='last')]
            # Resample to 15-min BEFORE joining (matches load_all_data flow)
            nc_h_15 = nc_h[['predicted_error']].rename(
                columns={'predicted_error': f'nowcast_h{h}_pred'}).resample('15min').ffill()
            nowcast = nowcast.join(nc_h_15, how='left')

    # Imbalance labels
    imb = pd.read_csv(REPO_ROOT / "data" / "master" / "master_imbalance_data.csv",
                      parse_dates=['datetime']).set_index('datetime').sort_index()
    imb = imb.rename(columns={
        'System Imbalance (MWh)': 'imbalance_mwh',
        'Imbalance Settlement Price (EUR/MWh)': 'imb_settle_price',
    })
    imb = imb[['imbalance_mwh', 'imb_settle_price']]

    return {
        'regulation_3min': reg,
        'load_3min': load,
        'production_3min': prod,
        'export_import_3min': xi,
        'solar_hourly': solar,
        'damas_load': damas,
        'da_prices': da,
        'market_spreads': mkt,
        'weather': weather,
        'nowcast_oos': nowcast,
        'imbalance': imb,
    }


def main():
    print("=" * 70)
    print("VERIFICATION: predict.py vs training pipeline")
    print("=" * 70)

    TRAIN_END = '2026-01-31'

    # ===== STEP 1: Run the training pipeline (load_all_data + build_features) =====
    print("\n[*] Running training pipeline (load_all_data + build_features)...")
    data = load_all_data()
    tml.TRAIN_END = TRAIN_END
    tml.TEST_START = TRAIN_END
    train_df, all_feature_cols = build_features(data, 8)
    print(f"[+] Training pipeline: {len(train_df)} rows, {len(all_feature_cols)} features")

    # ===== STEP 2: Run production pipeline (predict._prepare_data_dict + build_features) =====
    print("\n[*] Running production pipeline (predict.py)...")
    raw = load_raw_dataframes()

    # Create predictor without a model (we only need feature engineering)
    predictor = SpreadPredictor.__new__(SpreadPredictor)
    predictor.model = None
    predictor.train_end = TRAIN_END

    prod_data = predictor._prepare_data_dict(
        raw['regulation_3min'], raw['load_3min'], raw['production_3min'],
        raw['export_import_3min'], raw['solar_hourly'], raw['damas_load'],
        raw['da_prices'], raw['market_spreads'], raw['weather'],
        raw['nowcast_oos'], raw['imbalance'],
    )

    tml.TRAIN_END = TRAIN_END
    tml.TEST_START = TRAIN_END
    prod_df, _ = build_features(prod_data, 8)
    print(f"[+] Production pipeline: {len(prod_df)} rows")

    # ===== STEP 4: Compare feature values =====
    print("\n[*] Comparing features on overlapping rows...")

    # Find overlapping index
    overlap = train_df.index.intersection(prod_df.index)
    print(f"[+] Overlapping rows: {len(overlap)}")

    if len(overlap) == 0:
        print("[!] No overlapping rows to compare!")
        return

    # Compare each of the 52 selected features
    n_checked = 0
    n_mismatches = 0
    mismatch_details = []

    for feat in SELECTED_FEATURES:
        if feat not in train_df.columns:
            print(f"  [!] Feature '{feat}' not in training output")
            n_mismatches += 1
            continue
        if feat not in prod_df.columns:
            print(f"  [!] Feature '{feat}' not in production output")
            n_mismatches += 1
            continue

        train_vals = train_df.loc[overlap, feat]
        prod_vals = prod_df.loc[overlap, feat]

        # Both NaN = match. One NaN = mismatch. Both numeric = compare with tolerance.
        both_nan = train_vals.isna() & prod_vals.isna()
        one_nan = train_vals.isna() ^ prod_vals.isna()
        both_valid = train_vals.notna() & prod_vals.notna()

        nan_mismatches = one_nan.sum()

        if both_valid.sum() > 0:
            diff = (train_vals[both_valid] - prod_vals[both_valid]).abs()
            max_diff = diff.max()
            mean_diff = diff.mean()
            # Relative tolerance: allow 1e-10 absolute or 1e-8 relative
            rel_diff = diff / train_vals[both_valid].abs().clip(lower=1e-10)
            max_rel = rel_diff.max()
            n_numeric_mismatch = (diff > 1e-10).sum()
        else:
            max_diff = 0
            mean_diff = 0
            max_rel = 0
            n_numeric_mismatch = 0

        total_mismatch = nan_mismatches + n_numeric_mismatch
        n_checked += 1

        status = "OK" if total_mismatch == 0 else "MISMATCH"
        if total_mismatch > 0:
            n_mismatches += 1
            mismatch_details.append({
                'feature': feat,
                'nan_mismatches': int(nan_mismatches),
                'numeric_mismatches': int(n_numeric_mismatch),
                'max_abs_diff': float(max_diff),
                'max_rel_diff': float(max_rel),
                'valid_count': int(both_valid.sum()),
            })
            print(f"  [{status}] {feat:<35s}  nan_diff={nan_mismatches}, "
                  f"num_diff={n_numeric_mismatch}, max_abs={max_diff:.2e}, max_rel={max_rel:.2e}")
        else:
            valid_ct = both_valid.sum()
            nan_ct = both_nan.sum()
            print(f"  [{status}]   {feat:<35s}  {valid_ct} values match, {nan_ct} both-NaN")

    # ===== STEP 5: Summary =====
    print(f"\n{'='*70}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*70}")
    print(f"Features checked:   {n_checked}/{len(SELECTED_FEATURES)}")
    print(f"Features matching:  {n_checked - n_mismatches}")
    print(f"Features mismatched: {n_mismatches}")

    if n_mismatches == 0:
        print("\n[+] PASS: Production pipeline matches training pipeline exactly.")
    else:
        print(f"\n[!] FAIL: {n_mismatches} feature(s) have mismatches:")
        for d in mismatch_details:
            print(f"    {d['feature']}: {d['nan_mismatches']} NaN + {d['numeric_mismatches']} numeric "
                  f"(max_abs={d['max_abs_diff']:.2e}, max_rel={d['max_rel_diff']:.2e}, "
                  f"n_valid={d['valid_count']})")

    # ===== STEP 6: Spot-check a specific row =====
    print(f"\n{'='*70}")
    print("SPOT CHECK: Last overlapping row")
    print(f"{'='*70}")
    spot_ts = overlap[-1]
    print(f"Timestamp: {spot_ts}")
    for feat in SELECTED_FEATURES[:10]:
        tv = train_df.loc[spot_ts, feat] if feat in train_df.columns else 'N/A'
        pv = prod_df.loc[spot_ts, feat] if feat in prod_df.columns else 'N/A'
        match = "=" if (pd.isna(tv) and pd.isna(pv)) or (tv == pv) else "!="
        print(f"  {feat:<35s}  train={tv!s:>15s}  prod={pv!s:>15s}  {match}")

    print("\n... (showing first 10 of 52)")

    # ===== STEP 7: Type check =====
    print(f"\n{'='*70}")
    print("TYPE CHECK")
    print(f"{'='*70}")
    type_mismatches = 0
    for feat in SELECTED_FEATURES:
        if feat in train_df.columns and feat in prod_df.columns:
            t_dtype = train_df[feat].dtype
            p_dtype = prod_df[feat].dtype
            if t_dtype != p_dtype:
                print(f"  [!] {feat}: train={t_dtype}, prod={p_dtype}")
                type_mismatches += 1
    if type_mismatches == 0:
        print("[+] All feature dtypes match.")
    else:
        print(f"[!] {type_mismatches} dtype mismatches.")


if __name__ == "__main__":
    main()
