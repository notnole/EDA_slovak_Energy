"""
2-Hour-Ahead Imbalance Predictor — V2 (100 features)
=====================================================

Expands the original 55-feature model to ~100 features by incorporating
every available data source in the repository:

  NEW data sources:
    - Production SCADA (3-min)          -> 5 features
    - Export/Import SCADA (3-min)       -> 5 features
    - DAMAS load forecast error (1h)    -> 5 features
    - DA prices & cross-border (1h)     -> 7 features
    - Market spreads: IDM/Imb (1h)      -> 6 features
    - Temperature (15-min)              -> 4 features
    - Load nowcast H+2 predictions (1h) -> 4 features
    - Enhanced proxy derived            -> 6 features
    - Enhanced load derived             -> 4 features
    - Enhanced time                     -> 3 features

  Short-coverage features (production, export, temperature) are NaN
  for earlier periods — LightGBM handles NaN natively.

Target: System Imbalance (MWh) per 15-min settlement period
Lead:   8 settlement periods = 2 hours
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "plots_v2"
MODEL_DIR = BASE_DIR / "models_v2"
REPO_ROOT = SCRIPT_DIR.parents[2]

for d in [DATA_DIR, PLOT_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (16, 10), "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3,
})

LEAD_PERIODS = 8  # 8 x 15min = 2 hours
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
TRAIN_END = '2025-09-30'
TEST_START = '2025-10-01'


# ============================================================
# DATA LOADING
# ============================================================

def load_regulation():
    """Load 3-min regulation data and resample to 15-min."""
    print("[*] Loading regulation data...")
    path = REPO_ROOT / "data" / "features" / "regulation_3min.csv"
    df = pd.read_csv(path, parse_dates=['datetime'])
    df = df.set_index('datetime').sort_index()

    reg_15 = df['regulation_mw'].resample('15min').agg(['mean', 'std', 'min', 'max', 'count'])
    reg_15.columns = ['reg_mean', 'reg_std', 'reg_min', 'reg_max', 'reg_count']
    reg_15 = reg_15[reg_15['reg_count'] >= 2]
    reg_15['proxy'] = -0.25 * reg_15['reg_mean']

    print(f"[+] Regulation: {len(reg_15)} periods, {reg_15.index.min()} to {reg_15.index.max()}")
    return reg_15


def load_load_data():
    """Load 3-min load data and resample to 15-min."""
    print("[*] Loading load data...")
    path = REPO_ROOT / "data" / "features" / "load_3min.csv"
    df = pd.read_csv(path, parse_dates=['datetime'])
    df = df.set_index('datetime').sort_index()

    load_15 = df['load_mw'].resample('15min').agg(['mean', 'std', 'min', 'max'])
    load_15.columns = ['load_mean', 'load_std', 'load_min', 'load_max']

    print(f"[+] Load: {len(load_15)} periods")
    return load_15


def load_production():
    """Load 3-min production data and resample to 15-min. Short coverage (Oct 2025+)."""
    print("[*] Loading production data...")
    path = REPO_ROOT / "data" / "features" / "production_3min.csv"
    if not path.exists():
        print("[-] Production data not found, skipping")
        return None
    df = pd.read_csv(path, parse_dates=['datetime'])
    df = df.set_index('datetime').sort_index()

    prod_15 = df['production_mw'].resample('15min').agg(['mean', 'std'])
    prod_15.columns = ['prod_mean', 'prod_std']

    print(f"[+] Production: {len(prod_15)} periods, {prod_15.index.min()} to {prod_15.index.max()}")
    return prod_15


def load_export_import():
    """Load 3-min export/import data and resample to 15-min. Short coverage (Oct 2025+)."""
    print("[*] Loading export/import data...")
    path = REPO_ROOT / "data" / "features" / "export_import_3min.csv"
    if not path.exists():
        print("[-] Export/import data not found, skipping")
        return None
    df = pd.read_csv(path, parse_dates=['datetime'])
    df = df.set_index('datetime').sort_index()

    xi_15 = df['export_import_mw'].resample('15min').agg(['mean', 'std'])
    xi_15.columns = ['xborder_mean', 'xborder_std']

    print(f"[+] Export/Import: {len(xi_15)} periods, {xi_15.index.min()} to {xi_15.index.max()}")
    return xi_15


def load_solar():
    """Load hourly solar data, broadcast to 15-min."""
    print("[*] Loading solar data...")
    path = REPO_ROOT / "data" / "clean" / "solar" / "solar_hourly.csv"
    if not path.exists():
        print("[-] Solar data not found, skipping")
        return None
    df = pd.read_csv(path, parse_dates=['datetime'])
    df = df.set_index('datetime').sort_index()
    df = _dedup_index(df)
    solar_15 = df.resample('15min').ffill()
    print(f"[+] Solar: {len(solar_15)} periods")
    return solar_15


def _dedup_index(df):
    """Remove duplicate timestamps (DST transitions), keep last."""
    return df[~df.index.duplicated(keep='last')]


def load_damas_load():
    """Load DAMAS load forecast vs actual (hourly), broadcast to 15-min."""
    print("[*] Loading DAMAS load forecast data...")
    path = REPO_ROOT / "features" / "DamasLoad" / "load_data.csv"
    if not path.exists():
        print("[-] DAMAS load data not found, skipping")
        return None
    df = pd.read_csv(path, parse_dates=['datetime'])
    df = df.set_index('datetime').sort_index()
    df = _dedup_index(df)
    # Keep only the columns we need
    damas = df[['forecast_error_mw', 'forecast_error_pct', 'forecast_load_mw']].copy()
    # Broadcast hourly -> 15-min
    damas_15 = damas.resample('15min').ffill()
    print(f"[+] DAMAS load: {len(damas_15)} periods, {damas_15.index.min()} to {damas_15.index.max()}")
    return damas_15


def load_da_prices():
    """Load DA prices and cross-border flows (hourly), broadcast to 15-min.
    DA data is known D-1 at 11:00, so available for any intraday prediction."""
    print("[*] Loading DA price data...")
    path = REPO_ROOT / "features" / "DamasPrices" / "data" / "da_prices.csv"
    if not path.exists():
        print("[-] DA prices not found, skipping")
        return None
    df = pd.read_csv(path, parse_dates=['datetime'])
    df = df.set_index('datetime').sort_index()
    keep = ['price_eur_mwh', 'demand_mw', 'supply_mw',
            'net_flow_cz', 'net_flow_pl', 'net_flow_hu', 'net_import',
            'price_lag24', 'price_change_24h']
    keep = [c for c in keep if c in df.columns]
    da = df[keep].copy()
    da = _dedup_index(da)
    da_15 = da.resample('15min').ffill()
    print(f"[+] DA prices: {len(da_15)} periods")
    return da_15


def load_market_spreads():
    """Load hourly market prices with IDM/Imbalance spreads, broadcast to 15-min.
    IDM and imbalance prices need lagging (not known until after settlement)."""
    print("[*] Loading market spread data...")
    path = REPO_ROOT / "MarketPriceGap" / "data" / "processed" / "hourly_market_prices.csv"
    if not path.exists():
        print("[-] Market spreads not found, skipping")
        return None
    df = pd.read_csv(path, parse_dates=['timestamp_hour'])
    df = df.set_index('timestamp_hour').sort_index()
    keep = ['idm_vwap', 'idm_volume_mwh', 'imb_settlement_price',
            'spread_da_idm', 'spread_idm_imb', 'spread_da_imb']
    keep = [c for c in keep if c in df.columns]
    mkt = df[keep].copy()
    mkt = _dedup_index(mkt)
    mkt_15 = mkt.resample('15min').ffill()
    print(f"[+] Market spreads: {len(mkt_15)} periods")
    return mkt_15


def load_temperature():
    """Load 15-min temperature data. Short coverage (Sep 2025+)."""
    print("[*] Loading temperature data...")
    path = REPO_ROOT / "data" / "clean" / "weather" / "temperature_15min.csv"
    if not path.exists():
        print("[-] Temperature data not found, skipping")
        return None
    df = pd.read_csv(path, parse_dates=['datetime'])
    df = df.set_index('datetime').sort_index()
    # Use actual temperature; forecast as fallback
    temp = pd.DataFrame(index=df.index)
    if 'temp_actual' in df.columns:
        temp['temperature'] = df['temp_actual']
    elif 'temperature_2m' in df.columns:
        temp['temperature'] = df['temperature_2m']
    else:
        # Try first numeric column
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            temp['temperature'] = df[num_cols[0]]
        else:
            print("[-] No usable temperature column found")
            return None
    # Fill gaps from forecast if available
    if 'temp_forecast_gfs' in df.columns:
        temp['temperature'] = temp['temperature'].fillna(df['temp_forecast_gfs'])
    print(f"[+] Temperature: {len(temp)} periods, {temp.index.min()} to {temp.index.max()}")
    return temp


def load_load_nowcast():
    """Load H+2 load nowcast OOS predictions (hourly), broadcast to 15-min.
    Uses walk-forward out-of-sample predictions where each prediction was
    made by a model trained only on strictly prior data."""
    print("[*] Loading load nowcast H+2 OOS predictions...")
    # Prefer walk-forward OOS predictions (no leakage)
    oos_path = REPO_ROOT / "LoadAnalysis" / "nowcast_5h" / "tuning" / "oos_predictions" / "h2_oos_predictions.csv"
    if oos_path.exists():
        path = oos_path
        print("    Using walk-forward OOS predictions (clean)")
    else:
        print("[-] OOS predictions not found. Run generate_oos_predictions.py first.")
        print("    Falling back to DISABLED (NaN) to avoid leakage.")
        return None
    df = pd.read_csv(path, parse_dates=['datetime'])
    df = df.set_index('datetime').sort_index()
    df = _dedup_index(df)
    nowcast = df[['predicted_error', 'actual_error']].copy()
    nowcast.columns = ['nowcast_pred_error', 'nowcast_actual_error']
    nowcast_15 = nowcast.resample('15min').ffill()
    print(f"[+] Load nowcast: {len(nowcast_15)} periods, {nowcast_15.index.min()} to {nowcast_15.index.max()}")
    return nowcast_15


def load_imbalance():
    """Load 15-min imbalance labels."""
    print("[*] Loading imbalance labels...")
    path = REPO_ROOT / "data" / "master" / "master_imbalance_data.csv"
    df = pd.read_csv(path, parse_dates=['datetime'])
    df = df.set_index('datetime').sort_index()
    df = df.rename(columns={
        'System Imbalance (MWh)': 'imbalance_mwh',
        'Imbalance Settlement Price (EUR/MWh)': 'imb_settle_price',
    })
    print(f"[+] Imbalance: {len(df)} periods, {df.index.min()} to {df.index.max()}")
    return df[['imbalance_mwh', 'imb_settle_price']]


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(reg, load, prod, xborder, solar, damas_load, da_prices,
                   market_spreads, temperature, load_nowcast, imb):
    """
    Build ~100 feature matrix. Each row = one 15-min settlement period.
    All features use only data available LEAD_PERIODS (8) periods before.
    Short-coverage features are left as NaN (LightGBM handles natively).
    """
    print("\n[*] Building features...")

    # ------------------------------------------------------------------
    # MERGE all 15-min data (left-join new sources onto core)
    # ------------------------------------------------------------------
    df = reg.join(load, how='inner')
    if prod is not None:
        df = df.join(prod, how='left')
    if xborder is not None:
        df = df.join(xborder, how='left')
    if solar is not None:
        df = df.join(solar, how='left')
    if damas_load is not None:
        df = df.join(damas_load, how='left')
    if da_prices is not None:
        df = df.join(da_prices, how='left')
    if market_spreads is not None:
        df = df.join(market_spreads, how='left')
    if temperature is not None:
        df = df.join(temperature, how='left')
    if load_nowcast is not None:
        df = df.join(load_nowcast, how='left')
    df = df.join(imb, how='inner')

    # Temp columns for grouping
    df['hour'] = df.index.hour
    df['qh'] = df.index.minute // 15
    df['hour_qh'] = df['hour'] * 4 + df['qh']
    df['dow'] = df.index.dayofweek
    train_mask = df.index <= TRAIN_END
    test_mask = df.index > TRAIN_END

    features = {}  # collect all feature Series here

    # ==================================================================
    # GROUP 1: PROXY FEATURES (16 lags + 20 derived = 36 features)
    # ==================================================================
    # --- 1a. Raw lags (16) ---
    for lag in range(LEAD_PERIODS, LEAD_PERIODS + 16):
        features[f'proxy_lag{lag}'] = df['proxy'].shift(lag)

    # --- 1b. Rolling stats (9) ---
    proxy_s = df['proxy'].shift(LEAD_PERIODS)
    features['proxy_rmean4'] = proxy_s.rolling(4).mean()
    features['proxy_rmean8'] = proxy_s.rolling(8).mean()
    features['proxy_rmean16'] = proxy_s.rolling(16).mean()
    features['proxy_rmean32'] = proxy_s.rolling(32).mean()
    features['proxy_rstd4'] = proxy_s.rolling(4).std()
    features['proxy_rstd8'] = proxy_s.rolling(8).std()
    features['proxy_rmin4'] = proxy_s.rolling(4).min()
    features['proxy_rmax4'] = proxy_s.rolling(4).max()
    features['proxy_range4'] = features['proxy_rmax4'] - features['proxy_rmin4']

    # --- 1c. Momentum (3) ---
    features['proxy_momentum'] = features[f'proxy_lag{LEAD_PERIODS}'] - features[f'proxy_lag{LEAD_PERIODS+1}']
    features['proxy_momentum4'] = features[f'proxy_lag{LEAD_PERIODS}'] - features[f'proxy_lag{LEAD_PERIODS+4}']
    features['proxy_acceleration'] = features['proxy_momentum'] - (
        features[f'proxy_lag{LEAD_PERIODS+1}'] - features[f'proxy_lag{LEAD_PERIODS+2}']
    )

    # --- 1d. Sign-asymmetric (2) ---
    features['proxy_lag8_pos'] = features[f'proxy_lag{LEAD_PERIODS}'].clip(lower=0)
    features['proxy_lag8_neg'] = features[f'proxy_lag{LEAD_PERIODS}'].clip(upper=0)

    # --- 1e. Direction ratios (3) ---
    for w in [4, 8, 16]:
        pos_count = pd.Series(0.0, index=df.index)
        for i in range(LEAD_PERIODS, LEAD_PERIODS + w):
            pos_count += (df['proxy'].shift(i) > 0).astype(float)
        features[f'proxy_pos_ratio_{w}'] = pos_count / w

    # --- 1f. Yesterday (2) ---
    features['proxy_yesterday'] = df['proxy'].shift(96)
    features['proxy_yesterday_2'] = df['proxy'].shift(96 * 2)

    # --- 1g. Deviation from hourly baseline (1) ---
    proxy_train = proxy_s[train_mask]
    frozen_proxy_mean = proxy_train.groupby(df.loc[train_mask, 'hour_qh']).mean()
    proxy_baseline = proxy_train.groupby(df.loc[train_mask, 'hour_qh']).expanding().mean()
    proxy_baseline = proxy_baseline.droplevel(0).sort_index()
    proxy_baseline = proxy_baseline[~proxy_baseline.index.duplicated(keep='last')]
    proxy_bl = proxy_baseline.reindex(df.index)
    proxy_bl[test_mask] = df.loc[test_mask, 'hour_qh'].map(frozen_proxy_mean).values
    features['proxy_dev_from_hour'] = proxy_s - proxy_bl

    # ==================================================================
    # GROUP 2: ENHANCED PROXY DERIVED (6 new features)
    # ==================================================================
    features['proxy_ewm4'] = proxy_s.ewm(span=4).mean()
    features['proxy_range8'] = proxy_s.rolling(8).max() - proxy_s.rolling(8).min()
    # Sign changes in last 4 periods (volatility/choppiness)
    proxy_sign = np.sign(proxy_s)
    features['proxy_zero_cross4'] = sum(
        (np.sign(df['proxy'].shift(LEAD_PERIODS + i)) != np.sign(df['proxy'].shift(LEAD_PERIODS + i + 1))).astype(float)
        for i in range(4)
    )
    # Mean absolute proxy (magnitude regardless of sign)
    features['proxy_abs_rmean4'] = proxy_s.abs().rolling(4).mean()
    features['proxy_abs_rmean8'] = proxy_s.abs().rolling(8).mean()
    # Change vs same period yesterday
    features['proxy_lag96_diff'] = features[f'proxy_lag{LEAD_PERIODS}'] - df['proxy'].shift(96)

    # ==================================================================
    # GROUP 3: REGULATION FEATURES (5 original)
    # ==================================================================
    reg_s = df['reg_mean'].shift(LEAD_PERIODS)
    features['reg_rmean4'] = reg_s.rolling(4).mean()
    features['reg_rmean8'] = reg_s.rolling(8).mean()
    features['reg_rstd8'] = reg_s.rolling(8).std()
    features['reg_momentum'] = df['reg_mean'].shift(LEAD_PERIODS) - df['reg_mean'].shift(LEAD_PERIODS + 1)
    reg_std_s = df['reg_std'].shift(LEAD_PERIODS)
    features['reg_vol_rmean4'] = reg_std_s.rolling(4).mean()

    # ==================================================================
    # GROUP 4: LOAD FEATURES (4 original + 4 new = 8)
    # ==================================================================
    load_s = df['load_mean'].shift(LEAD_PERIODS)

    features['load_rmean4'] = load_s.rolling(4).mean()
    features['load_rmean8'] = load_s.rolling(8).mean()      # NEW
    features['load_rmean16'] = load_s.rolling(16).mean()
    features['load_momentum'] = df['load_mean'].shift(LEAD_PERIODS) - df['load_mean'].shift(LEAD_PERIODS + 4)

    # Load deviation from hourly baseline (leakage-safe)
    load_train = load_s[train_mask]
    frozen_load_mean = load_train.groupby(df.loc[train_mask, 'hour_qh']).mean()
    load_bl_expanding = load_train.groupby(df.loc[train_mask, 'hour_qh']).expanding().mean()
    load_bl_expanding = load_bl_expanding.droplevel(0).sort_index()
    load_bl_expanding = load_bl_expanding[~load_bl_expanding.index.duplicated(keep='last')]
    load_bl = load_bl_expanding.reindex(df.index)
    load_bl[test_mask] = df.loc[test_mask, 'hour_qh'].map(frozen_load_mean).values
    features['load_deviation'] = load_s - load_bl

    # NEW: load std, range, yesterday
    features['load_rstd4'] = load_s.rolling(4).std()         # NEW
    load_min_s = df['load_min'].shift(LEAD_PERIODS) if 'load_min' in df.columns else load_s
    load_max_s = df['load_max'].shift(LEAD_PERIODS) if 'load_max' in df.columns else load_s
    features['load_ramp4'] = load_max_s.rolling(4).max() - load_min_s.rolling(4).min()  # NEW
    features['load_yesterday'] = df['load_mean'].shift(96)   # NEW

    # ==================================================================
    # GROUP 5: PRODUCTION SCADA (5 features, NaN pre-Oct 2025)
    # ==================================================================
    if 'prod_mean' in df.columns:
        prod_s = df['prod_mean'].shift(LEAD_PERIODS)
        features['prod_rmean4'] = prod_s.rolling(4).mean()
        features['prod_rmean8'] = prod_s.rolling(8).mean()
        features['prod_momentum'] = df['prod_mean'].shift(LEAD_PERIODS) - df['prod_mean'].shift(LEAD_PERIODS + 4)
        # Production deviation from hourly baseline (use train subset where available)
        prod_train = prod_s[train_mask].dropna()
        if len(prod_train) > 96:
            frozen_prod_mean = prod_train.groupby(df.loc[prod_train.index, 'hour_qh']).mean()
            features['prod_deviation'] = prod_s - df['hour_qh'].map(frozen_prod_mean)
        else:
            features['prod_deviation'] = pd.Series(np.nan, index=df.index)
        # Production volatility
        prod_std_s = df['prod_std'].shift(LEAD_PERIODS) if 'prod_std' in df.columns else pd.Series(np.nan, index=df.index)
        features['prod_vol'] = prod_std_s.rolling(4).mean()
    else:
        for f in ['prod_rmean4', 'prod_rmean8', 'prod_momentum', 'prod_deviation', 'prod_vol']:
            features[f] = pd.Series(np.nan, index=df.index)

    # ==================================================================
    # GROUP 6: EXPORT/IMPORT SCADA (5 features, NaN pre-Oct 2025)
    # ==================================================================
    if 'xborder_mean' in df.columns:
        xb_s = df['xborder_mean'].shift(LEAD_PERIODS)
        features['xborder_rmean4'] = xb_s.rolling(4).mean()
        features['xborder_rmean8'] = xb_s.rolling(8).mean()
        features['xborder_momentum'] = df['xborder_mean'].shift(LEAD_PERIODS) - df['xborder_mean'].shift(LEAD_PERIODS + 4)
        xb_train = xb_s[train_mask].dropna()
        if len(xb_train) > 96:
            frozen_xb_mean = xb_train.groupby(df.loc[xb_train.index, 'hour_qh']).mean()
            features['xborder_deviation'] = xb_s - df['hour_qh'].map(frozen_xb_mean)
        else:
            features['xborder_deviation'] = pd.Series(np.nan, index=df.index)
        features['xborder_vol'] = (df['xborder_std'].shift(LEAD_PERIODS).rolling(4).mean()
                                   if 'xborder_std' in df.columns
                                   else pd.Series(np.nan, index=df.index))
    else:
        for f in ['xborder_rmean4', 'xborder_rmean8', 'xborder_momentum', 'xborder_deviation', 'xborder_vol']:
            features[f] = pd.Series(np.nan, index=df.index)

    # ==================================================================
    # GROUP 7: SOLAR FEATURES (2 original)
    # Solar actual for hour H is only known after H ends.
    # Shift by LEAD_PERIODS + 4 (3h) to match DAMAS treatment.
    # ==================================================================
    if solar is not None and 'solar_surprise_mw' in df.columns:
        solar_s = df['solar_surprise_mw'].shift(LEAD_PERIODS + 4)
        features['solar_surprise_lag'] = solar_s
        features['solar_surprise_rmean4'] = solar_s.rolling(4).mean()
    else:
        features['solar_surprise_lag'] = 0
        features['solar_surprise_rmean4'] = 0

    # ==================================================================
    # GROUP 8: DAMAS LOAD FORECAST ERROR (5 features, full coverage)
    # ==================================================================
    if 'forecast_error_mw' in df.columns:
        # DAMAS forecast is known D-1; actual becomes known after the hour.
        # At T-2h, the most recent actual hour ended ~2-3h ago.
        # Shift by LEAD_PERIODS + 4 (3h total) to be safe.
        fe_s = df['forecast_error_mw'].shift(LEAD_PERIODS + 4)
        features['damas_fe'] = fe_s
        features['damas_fe_abs'] = fe_s.abs()
        features['damas_fe_rmean4'] = fe_s.rolling(4).mean()
        features['damas_fe_rmean24'] = fe_s.rolling(24 * 4).mean()  # 24h rolling (96 periods)
        # DAMAS forecast load for the target hour (known D-1, no lag needed)
        features['damas_forecast_load'] = df['forecast_load_mw']
    else:
        for f in ['damas_fe', 'damas_fe_abs', 'damas_fe_rmean4', 'damas_fe_rmean24', 'damas_forecast_load']:
            features[f] = pd.Series(np.nan, index=df.index)

    # ==================================================================
    # GROUP 9: DA PRICES & CROSS-BORDER (7 features, full coverage)
    # DA data known D-1 at 11:00 — no lag needed for intraday use.
    # ==================================================================
    if 'price_eur_mwh' in df.columns:
        features['da_price'] = df['price_eur_mwh']
        features['da_price_change24h'] = df['price_change_24h'] if 'price_change_24h' in df.columns else pd.Series(np.nan, index=df.index)
        features['da_demand'] = df['demand_mw'] if 'demand_mw' in df.columns else pd.Series(np.nan, index=df.index)
        features['da_supply'] = df['supply_mw'] if 'supply_mw' in df.columns else pd.Series(np.nan, index=df.index)
        features['da_net_import'] = df['net_import'] if 'net_import' in df.columns else pd.Series(np.nan, index=df.index)
        # Cross-border flows by neighbor (CZ most important for SK)
        features['da_flow_cz'] = df['net_flow_cz'] if 'net_flow_cz' in df.columns else pd.Series(np.nan, index=df.index)
        features['da_flow_hu'] = df['net_flow_hu'] if 'net_flow_hu' in df.columns else pd.Series(np.nan, index=df.index)
    else:
        for f in ['da_price', 'da_price_change24h', 'da_demand', 'da_supply',
                   'da_net_import', 'da_flow_cz', 'da_flow_hu']:
            features[f] = pd.Series(np.nan, index=df.index)

    # ==================================================================
    # GROUP 10: MARKET SPREADS — IDM/IMB (6 features)
    # IDM and imbalance prices are only known after settlement.
    # Shift by LEAD_PERIODS + 4 (3h) to be conservative.
    # ==================================================================
    mkt_shift = LEAD_PERIODS + 4  # 3 hours of safety margin

    if 'idm_vwap' in df.columns:
        features['idm_vwap_lag'] = df['idm_vwap'].shift(mkt_shift)
        features['idm_volume_lag'] = df['idm_volume_mwh'].shift(mkt_shift) if 'idm_volume_mwh' in df.columns else pd.Series(np.nan, index=df.index)
        features['spread_da_idm_lag'] = df['spread_da_idm'].shift(mkt_shift) if 'spread_da_idm' in df.columns else pd.Series(np.nan, index=df.index)
    else:
        features['idm_vwap_lag'] = pd.Series(np.nan, index=df.index)
        features['idm_volume_lag'] = pd.Series(np.nan, index=df.index)
        features['spread_da_idm_lag'] = pd.Series(np.nan, index=df.index)

    if 'imb_settle_price' in df.columns:
        features['imb_price_lag'] = df['imb_settle_price'].shift(mkt_shift)
        features['imb_price_rmean4'] = df['imb_settle_price'].shift(mkt_shift).rolling(4).mean()
        features['spread_da_imb_lag'] = (df['price_eur_mwh'] - df['imb_settle_price'].shift(mkt_shift)
                                         if 'price_eur_mwh' in df.columns
                                         else pd.Series(np.nan, index=df.index))
    else:
        features['imb_price_lag'] = pd.Series(np.nan, index=df.index)
        features['imb_price_rmean4'] = pd.Series(np.nan, index=df.index)
        features['spread_da_imb_lag'] = pd.Series(np.nan, index=df.index)

    # ==================================================================
    # GROUP 11: TEMPERATURE (4 features, NaN pre-Sep 2025)
    # ==================================================================
    if 'temperature' in df.columns:
        temp_s = df['temperature'].shift(LEAD_PERIODS)
        features['temp'] = temp_s
        features['temp_change6h'] = temp_s - df['temperature'].shift(LEAD_PERIODS + 24)  # 6h = 24 periods
        features['temp_rmean24h'] = temp_s.rolling(96).mean()
        # Temperature deviation from rolling 7-day mean (seasonal anomaly)
        features['temp_deviation'] = temp_s - temp_s.rolling(96 * 7, min_periods=96).mean()
    else:
        for f in ['temp', 'temp_change6h', 'temp_rmean24h', 'temp_deviation']:
            features[f] = pd.Series(np.nan, index=df.index)

    # ==================================================================
    # GROUP 12: LOAD NOWCAST H+2 PREDICTIONS (4 features)
    # Now using walk-forward OOS predictions (generate_oos_predictions.py).
    # The H+2 prediction was made 2h before the target hour — exactly
    # our prediction time. So we can use it directly without extra lag.
    # nowcast_recent_bias uses actual error which needs 3h shift (like DAMAS).
    # ==================================================================
    if 'nowcast_pred_error' in df.columns:
        features['nowcast_pred_error'] = df['nowcast_pred_error']
        features['nowcast_pred_error_abs'] = df['nowcast_pred_error'].abs()
        features['nowcast_pred_rmean4'] = df['nowcast_pred_error'].rolling(4).mean()
        # Meta-signal: how wrong was the nowcast recently? (lagged actual error)
        # Actual error needs same 3h shift as DAMAS forecast error
        nowcast_bias_s = (df['nowcast_pred_error'] - df['nowcast_actual_error']).shift(LEAD_PERIODS + 4)
        features['nowcast_recent_bias'] = nowcast_bias_s.rolling(8).mean()
    else:
        for f in ['nowcast_pred_error', 'nowcast_pred_error_abs', 'nowcast_pred_rmean4', 'nowcast_recent_bias']:
            features[f] = pd.Series(np.nan, index=df.index)

    # ==================================================================
    # GROUP 13: TIME FEATURES (6 original + 3 new = 9)
    # ==================================================================
    features['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    features['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    features['qh_in_hour'] = df['qh']
    features['is_weekend'] = (df['dow'] >= 5).astype(int)
    features['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    features['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    # NEW time features
    features['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    features['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
    features['is_peak'] = ((df['hour'] >= 7) & (df['hour'] <= 22)).astype(int)

    # ==================================================================
    # ASSEMBLE
    # ==================================================================
    feat_df = pd.DataFrame(features, index=df.index)
    feature_cols = list(feat_df.columns)

    print(f"[+] Total features: {len(feature_cols)}")

    # Add target
    feat_df['target'] = df['imbalance_mwh']

    # Only drop rows where target or CORE features (proxy, load, time) are NaN.
    # Short-coverage features stay NaN — LightGBM handles them.
    core_cols = [f'proxy_lag{LEAD_PERIODS}', 'proxy_rmean4', 'load_rmean4',
                 'hour_sin', 'target']
    valid = feat_df.dropna(subset=core_cols)

    print(f"[+] Valid samples: {len(valid)} (dropped {len(feat_df) - len(valid)} with NaN in core)")

    # Report coverage of optional features
    print("\n[*] Feature coverage (non-NaN %):")
    groups = {
        'Production': ['prod_rmean4'],
        'Export/Import': ['xborder_rmean4'],
        'DAMAS FE': ['damas_fe'],
        'DA Price': ['da_price'],
        'IDM/Spread': ['idm_vwap_lag'],
        'Temperature': ['temp'],
        'Load Nowcast': ['nowcast_pred_error'],
    }
    for group, cols in groups.items():
        for c in cols:
            if c in valid.columns:
                pct = valid[c].notna().mean() * 100
                print(f"    {group:20s}: {pct:5.1f}% non-NaN")

    return valid, feature_cols


# ============================================================
# TRAINING
# ============================================================

def train_quantile_models(df, feature_cols):
    """Train LightGBM models for each quantile."""
    print("\n[*] Training quantile models...")

    train = df[df.index <= TRAIN_END]
    test = df[df.index >= TEST_START]

    print(f"    Train: {len(train)} samples ({train.index.min()} to {train.index.max()})")
    print(f"    Test:  {len(test)} samples ({test.index.min()} to {test.index.max()})")

    X_train = train[feature_cols].values
    y_train = train['target'].values
    X_test = test[feature_cols].values
    y_test = test['target'].values

    models = {}
    predictions = {}

    for q in QUANTILES:
        print(f"    Training quantile {q:.2f}...")

        params = {
            'objective': 'quantile',
            'alpha': q,
            'learning_rate': 0.05,
            'num_leaves': 63,
            'min_child_samples': 50,
            'subsample': 0.8,
            'colsample_bytree': 0.7,  # slightly lower for more features
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'n_estimators': 600,       # slightly more trees for more features
            'verbose': -1,
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.log_evaluation(0)],
        )

        pred = model.predict(X_test)
        predictions[f'q{int(q*100)}'] = pred
        models[q] = model

        joblib.dump(model, MODEL_DIR / f"imb_2h_v2_q{int(q*100)}.joblib")

    # Build predictions DataFrame
    pred_df = test[['target']].copy()
    for q_name, pred in predictions.items():
        pred_df[q_name] = pred

    pred_df['pred_median'] = pred_df['q50']
    pred_df['pred_direction'] = np.sign(pred_df['pred_median'])
    pred_df['actual_direction'] = np.sign(pred_df['target'])
    pred_df['confidence'] = pred_df['q90'] - pred_df['q10']

    return models, pred_df, feature_cols


# ============================================================
# EVALUATION
# ============================================================

def evaluate(pred_df, models, feature_cols, df):
    """Evaluate model performance."""
    print("\n" + "=" * 70)
    print("MODEL EVALUATION (V2 - 100 features)")
    print("=" * 70)

    y_true = pred_df['target'].values
    y_pred = pred_df['pred_median'].values

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    corr = np.corrcoef(y_true, y_pred)[0, 1]

    print(f"\n  Point prediction (median):")
    print(f"    MAE:  {mae:.2f} MWh")
    print(f"    RMSE: {rmse:.2f} MWh")
    print(f"    R2:   {r2:.3f}")
    print(f"    Corr: {corr:.3f}")

    # Direction accuracy
    dir_correct = (pred_df['pred_direction'] == pred_df['actual_direction'])
    nonzero = pred_df['target'].abs() > 0.1
    dir_acc = dir_correct[nonzero].mean()
    print(f"\n  Direction accuracy: {dir_acc:.1%} ({dir_correct[nonzero].sum()}/{nonzero.sum()} non-zero periods)")

    # Direction by confidence
    print("\n  Direction accuracy by confidence (80% CI width):")
    pred_df['conf_q'] = pd.qcut(pred_df['confidence'], q=5, labels=['Q1_narrow', 'Q2', 'Q3', 'Q4', 'Q5_wide'])
    for q in ['Q1_narrow', 'Q2', 'Q3', 'Q4', 'Q5_wide']:
        mask = (pred_df['conf_q'] == q) & nonzero
        sub = pred_df[mask]
        if len(sub) > 0:
            acc = (sub['pred_direction'] == sub['actual_direction']).mean()
            avg_ci = sub['confidence'].mean()
            print(f"    {q}: {acc:.1%} accuracy, avg CI = {avg_ci:.1f} MWh, n = {len(sub)}")

    # Direction by predicted magnitude
    print("\n  Direction accuracy by |predicted imbalance|:")
    pred_df['pred_abs'] = pred_df['pred_median'].abs()
    for lo, hi, label in [(0, 2, '0-2 MWh'), (2, 5, '2-5 MWh'), (5, 10, '5-10 MWh'),
                          (10, 20, '10-20 MWh'), (20, 999, '>20 MWh')]:
        mask = (pred_df['pred_abs'] >= lo) & (pred_df['pred_abs'] < hi) & nonzero
        sub = pred_df[mask]
        if len(sub) > 0:
            acc = (sub['pred_direction'] == sub['actual_direction']).mean()
            print(f"    |pred| {label}: {acc:.1%} ({len(sub)} periods)")

    # Quantile calibration
    print("\n  Quantile calibration:")
    for q in QUANTILES:
        q_col = f'q{int(q*100)}'
        actual_below = (y_true < pred_df[q_col].values).mean()
        print(f"    Q{int(q*100)}: target {q:.0%}, actual {actual_below:.1%}")

    # By hour
    pred_df['hour'] = pred_df.index.hour
    print("\n  Direction accuracy by hour:")
    hourly = pred_df[nonzero].groupby('hour').apply(
        lambda g: (g['pred_direction'] == g['actual_direction']).mean()
    )
    for h, acc in hourly.items():
        bar = '#' * int(acc * 50)
        print(f"    H{h:02d}: {acc:.0%} {bar}")

    # Feature importance
    print(f"\n  Top 25 features (median model, {len(feature_cols)} total):")
    imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': models[0.50].feature_importances_,
    }).sort_values('importance', ascending=False)
    imp['pct'] = imp['importance'] / imp['importance'].sum() * 100
    for _, r in imp.head(25).iterrows():
        print(f"    {r['feature']:<30} {r['pct']:>5.2f}%")

    # Report which new feature groups contribute most
    print("\n  Feature importance by group:")
    group_map = {
        'Proxy (original)': lambda f: f.startswith('proxy_lag') or f in [
            'proxy_rmean4', 'proxy_rmean8', 'proxy_rmean16', 'proxy_rmean32',
            'proxy_rstd4', 'proxy_rstd8', 'proxy_rmin4', 'proxy_rmax4', 'proxy_range4',
            'proxy_momentum', 'proxy_momentum4', 'proxy_acceleration',
            'proxy_lag8_pos', 'proxy_lag8_neg',
            'proxy_pos_ratio_4', 'proxy_pos_ratio_8', 'proxy_pos_ratio_16',
            'proxy_yesterday', 'proxy_yesterday_2', 'proxy_dev_from_hour'],
        'Proxy (new derived)': lambda f: f in [
            'proxy_ewm4', 'proxy_range8', 'proxy_zero_cross4',
            'proxy_abs_rmean4', 'proxy_abs_rmean8', 'proxy_lag96_diff'],
        'Regulation': lambda f: f.startswith('reg_'),
        'Load (original)': lambda f: f in [
            'load_rmean4', 'load_rmean16', 'load_momentum', 'load_deviation'],
        'Load (new)': lambda f: f in [
            'load_rmean8', 'load_rstd4', 'load_ramp4', 'load_yesterday'],
        'Production': lambda f: f.startswith('prod_'),
        'Export/Import': lambda f: f.startswith('xborder_'),
        'Solar': lambda f: f.startswith('solar_'),
        'DAMAS FE': lambda f: f.startswith('damas_'),
        'DA Price': lambda f: f.startswith('da_'),
        'Market/Spread': lambda f: f.startswith(('idm_', 'imb_price', 'spread_')),
        'Temperature': lambda f: f.startswith('temp'),
        'Load Nowcast': lambda f: f.startswith('nowcast_'),
        'Time': lambda f: f in [
            'hour_sin', 'hour_cos', 'qh_in_hour', 'is_weekend',
            'month_sin', 'month_cos', 'dow_sin', 'dow_cos', 'is_peak'],
    }
    for gname, gfunc in group_map.items():
        cols_in_group = [f for f in feature_cols if gfunc(f)]
        if cols_in_group:
            group_imp = imp[imp['feature'].isin(cols_in_group)]['pct'].sum()
            print(f"    {gname:25s}: {group_imp:5.1f}% ({len(cols_in_group)} features)")

    imp.to_csv(DATA_DIR / "feature_importance_v2.csv", index=False)
    pred_df.to_csv(DATA_DIR / "predictions_test_v2.csv")

    return pred_df, imp


# ============================================================
# VISUALIZATION
# ============================================================

def plot_results(pred_df, imp, feature_cols):
    """Generate evaluation plots."""

    # --- Plot 1: Pred vs actual + direction by magnitude + sample series ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    ax.scatter(pred_df['target'], pred_df['pred_median'], alpha=0.1, s=5, color='steelblue')
    lims = [-80, 80]
    ax.plot(lims, lims, 'r--', alpha=0.5)
    ax.set_xlabel("Actual Imbalance (MWh)")
    ax.set_ylabel("Predicted Imbalance (MWh)")
    ax.set_title("Predicted vs Actual (V2, 2h ahead)")
    ax.set_xlim(lims); ax.set_ylim(lims)

    ax = axes[1]
    bins = np.arange(0, 35, 2)
    pred_df['pred_abs_bin'] = pd.cut(pred_df['pred_median'].abs(), bins=bins)
    nonzero = pred_df['target'].abs() > 0.1
    dir_by_mag = pred_df[nonzero].groupby('pred_abs_bin', observed=True).apply(
        lambda g: (g['pred_direction'] == g['actual_direction']).mean()
    )
    counts = pred_df[nonzero].groupby('pred_abs_bin', observed=True).size()
    x = range(len(dir_by_mag))
    ax.bar(x, dir_by_mag.values, alpha=0.7, color='steelblue')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='50%')
    ax.axhline(0.65, color='orange', linestyle='--', alpha=0.5, label='65%')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{b.left:.0f}' for b in dir_by_mag.index], rotation=45)
    ax.set_xlabel("|Predicted Imbalance| (MWh)")
    ax.set_ylabel("Direction Accuracy")
    ax.set_title("Direction Accuracy by Magnitude")
    ax.legend(fontsize=9)
    for i, (acc, cnt) in enumerate(zip(dir_by_mag.values, counts.values)):
        ax.text(i, acc + 0.01, f'n={cnt}', ha='center', fontsize=7)

    ax = axes[2]
    sample = pred_df.iloc[:96*3]
    ax.fill_between(range(len(sample)), sample['q10'], sample['q90'], alpha=0.2, color='blue', label='P10-P90')
    ax.fill_between(range(len(sample)), sample['q25'], sample['q75'], alpha=0.3, color='blue', label='P25-P75')
    ax.plot(range(len(sample)), sample['pred_median'], 'b-', lw=1, label='Predicted')
    ax.plot(range(len(sample)), sample['target'], 'k-', lw=1, alpha=0.7, label='Actual')
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel("Settlement period"); ax.set_ylabel("Imbalance (MWh)")
    ax.set_title("3-Day Sample with Confidence Bands")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "01_model_evaluation_v2.png", dpi=150)
    plt.close()
    print(f"[+] Saved: {PLOT_DIR / '01_model_evaluation_v2.png'}")

    # --- Plot 2: Hourly direction accuracy ---
    fig, ax = plt.subplots(figsize=(12, 5))
    hourly_acc = pred_df[nonzero].groupby(pred_df[nonzero].index.hour).apply(
        lambda g: (g['pred_direction'] == g['actual_direction']).mean()
    )
    colors = ['#2ecc71' if a >= 0.65 else '#f39c12' if a >= 0.55 else '#e74c3c' for a in hourly_acc]
    ax.bar(hourly_acc.index, hourly_acc.values, color=colors, alpha=0.8)
    ax.axhline(0.65, color='orange', linestyle='--', label='65% target')
    ax.axhline(0.50, color='red', linestyle='--', label='50% coin flip')
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Direction accuracy")
    ax.set_title("Direction Accuracy by Hour (V2)")
    ax.set_xticks(range(24)); ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "02_hourly_direction_v2.png", dpi=150)
    plt.close()
    print(f"[+] Saved: {PLOT_DIR / '02_hourly_direction_v2.png'}")

    # --- Plot 3: Feature importance by group ---
    fig, ax = plt.subplots(figsize=(12, 6))
    top30 = imp.head(30)
    colors_map = {
        'proxy': '#3498db', 'reg': '#2ecc71', 'load': '#e67e22',
        'prod': '#9b59b6', 'xborder': '#1abc9c', 'solar': '#f1c40f',
        'damas': '#e74c3c', 'da_': '#34495e', 'idm': '#c0392b',
        'imb_': '#c0392b', 'spread': '#c0392b', 'temp': '#2980b9',
        'nowcast': '#d35400', 'hour': '#7f8c8d', 'qh': '#7f8c8d',
        'is_': '#7f8c8d', 'dow': '#7f8c8d', 'month': '#7f8c8d',
    }
    bar_colors = []
    for f in top30['feature']:
        c = '#95a5a6'
        for prefix, color in colors_map.items():
            if f.startswith(prefix):
                c = color
                break
        bar_colors.append(c)
    ax.barh(range(len(top30)-1, -1, -1), top30['pct'].values, color=bar_colors, alpha=0.8)
    ax.set_yticks(range(len(top30)-1, -1, -1))
    ax.set_yticklabels(top30['feature'].values, fontsize=9)
    ax.set_xlabel("Importance (%)")
    ax.set_title(f"Top 30 Feature Importance (of {len(feature_cols)} total)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "03_feature_importance_v2.png", dpi=150)
    plt.close()
    print(f"[+] Saved: {PLOT_DIR / '03_feature_importance_v2.png'}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("2-HOUR AHEAD IMBALANCE PREDICTOR — V2 (100 features)")
    print(f"Lead: {LEAD_PERIODS} periods ({LEAD_PERIODS * 15} minutes)")
    print(f"Train: up to {TRAIN_END}, Test: from {TEST_START}")
    print("=" * 70)

    # Load all data sources
    reg = load_regulation()
    load = load_load_data()
    prod = load_production()
    xborder = load_export_import()
    solar = load_solar()
    damas_load = load_damas_load()
    da_prices = load_da_prices()
    market_spreads = load_market_spreads()
    temperature = load_temperature()
    load_nowcast = load_load_nowcast()
    imb = load_imbalance()

    # Build features
    df, feature_cols = build_features(
        reg, load, prod, xborder, solar, damas_load, da_prices,
        market_spreads, temperature, load_nowcast, imb
    )

    # Train
    models, pred_df, feature_cols = train_quantile_models(df, feature_cols)

    # Evaluate
    pred_df, imp = evaluate(pred_df, models, feature_cols, df)

    # Plot
    plot_results(pred_df, imp, feature_cols)

    # Trading summary
    print("\n" + "=" * 70)
    print("TRADING RELEVANCE")
    print("=" * 70)

    nonzero = pred_df['target'].abs() > 0.1
    dir_acc = (pred_df.loc[nonzero, 'pred_direction'] == pred_df.loc[nonzero, 'actual_direction']).mean()
    print(f"\n  Overall direction accuracy: {dir_acc:.1%}")

    high_conf = pred_df['pred_abs'] > 5
    if high_conf.sum() > 0:
        hc_acc = (pred_df.loc[high_conf & nonzero, 'pred_direction'] == pred_df.loc[high_conf & nonzero, 'actual_direction']).mean()
        print(f"  High confidence (|pred|>5): {hc_acc:.1%} ({(high_conf & nonzero).sum()} periods, {(high_conf & nonzero).sum() / len(pred_df) * 100:.0f}% of all)")

    very_high = pred_df['pred_abs'] > 10
    if very_high.sum() > 0:
        vh_acc = (pred_df.loc[very_high & nonzero, 'pred_direction'] == pred_df.loc[very_high & nonzero, 'actual_direction']).mean()
        print(f"  Very high conf (|pred|>10): {vh_acc:.1%} ({(very_high & nonzero).sum()} periods)")

    print(f"\n  Periods per day (test): {len(pred_df) / pred_df.index.normalize().nunique():.0f}")
    print(f"  High-conf periods/day:  {high_conf.sum() / pred_df.index.normalize().nunique():.1f}")

    print("\n" + "=" * 70)
    print("[+] Done. V2 models -> models_v2/, predictions -> data/")
    print("=" * 70)


if __name__ == "__main__":
    main()
