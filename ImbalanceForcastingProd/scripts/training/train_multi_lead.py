"""
Multi-Lead Imbalance Predictor (Leads 4-8 = 1h to 2h ahead)
=============================================================

Trains the same 108-feature LightGBM model at 5 lead times:
  Lead 8 = 2h00 ahead (T-2h)    -- first signal
  Lead 7 = 1h45 ahead (T-1h45)
  Lead 6 = 1h30 ahead (T-1h30)
  Lead 5 = 1h15 ahead (T-1h15)
  Lead 4 = 1h00 ahead (T-1h)    -- most accurate

Shorter leads have access to more recent SCADA data, so accuracy improves.
The CASCADE STRATEGY uses the trajectory of predictions across leads:
  - If prediction strengthens (moves further from 0): increase position
  - If prediction weakens or flips: reduce/close position

Feature engineering is identical to train_imbalance_2h.py except:
  - SCADA shifts use the lead value (4-8) instead of fixed 8
  - Hourly data shifts use lead + 4 (1h publication delay, independent of lead)
  - DA data has no shift regardless of lead (known D-1)
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
BASE_DIR = SCRIPT_DIR.parents[1]  # ImbalanceForcastingProd/
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "plots"
MODEL_DIR = BASE_DIR / "models"
REPO_ROOT = BASE_DIR.parent  # repo root

for d in [DATA_DIR, PLOT_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (16, 10), "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3,
})

LEADS = [8, 7, 6, 5, 4]  # 2h, 1h45, 1h30, 1h15, 1h
QUANTILES = [0.10, 0.50, 0.90]  # Fewer quantiles for speed (5 leads x 3 = 15 models)
TRAIN_END = '2026-01-31'
TEST_START = '2026-02-01'


# ============================================================
# DATA LOADING (same as train_imbalance_2h.py)
# ============================================================

def _dedup_index(df):
    return df[~df.index.duplicated(keep='last')]


def load_all_data():
    """Load all data sources once, return as dict."""
    print("[*] Loading all data sources...")

    # Regulation
    path = REPO_ROOT / "data" / "features" / "regulation_3min.csv"
    reg = pd.read_csv(path, parse_dates=['datetime']).set_index('datetime').sort_index()
    reg_15 = reg['regulation_mw'].resample('15min').agg(['mean', 'std', 'min', 'max', 'count'])
    reg_15.columns = ['reg_mean', 'reg_std', 'reg_min', 'reg_max', 'reg_count']
    reg_15 = reg_15[reg_15['reg_count'] >= 2]
    reg_15['proxy'] = -0.25 * reg_15['reg_mean']
    print(f"[+] Regulation: {len(reg_15)} periods")

    # Load
    path = REPO_ROOT / "data" / "features" / "load_3min.csv"
    load = pd.read_csv(path, parse_dates=['datetime']).set_index('datetime').sort_index()
    load_15 = load['load_mw'].resample('15min').agg(['mean', 'std', 'min', 'max'])
    load_15.columns = ['load_mean', 'load_std', 'load_min', 'load_max']
    print(f"[+] Load: {len(load_15)} periods")

    # Production
    path = REPO_ROOT / "data" / "features" / "production_3min.csv"
    prod_15 = None
    if path.exists():
        prod = pd.read_csv(path, parse_dates=['datetime']).set_index('datetime').sort_index()
        prod_15 = prod['production_mw'].resample('15min').agg(['mean', 'std'])
        prod_15.columns = ['prod_mean', 'prod_std']
        print(f"[+] Production: {len(prod_15)} periods")

    # Export/Import
    path = REPO_ROOT / "data" / "features" / "export_import_3min.csv"
    xi_15 = None
    if path.exists():
        xi = pd.read_csv(path, parse_dates=['datetime']).set_index('datetime').sort_index()
        xi_15 = xi['export_import_mw'].resample('15min').agg(['mean', 'std'])
        xi_15.columns = ['xborder_mean', 'xborder_std']
        print(f"[+] Export/Import: {len(xi_15)} periods")

    # Solar
    path = REPO_ROOT / "data" / "clean" / "solar" / "solar_hourly.csv"
    solar_15 = None
    if path.exists():
        solar = pd.read_csv(path, parse_dates=['datetime']).set_index('datetime').sort_index()
        solar = _dedup_index(solar)
        solar_15 = solar.resample('15min').ffill()
        print(f"[+] Solar: {len(solar_15)} periods")

    # DAMAS load
    path = REPO_ROOT / "features" / "DamasLoad" / "load_data.csv"
    damas_15 = None
    if path.exists():
        damas = pd.read_csv(path, parse_dates=['datetime']).set_index('datetime').sort_index()
        damas = _dedup_index(damas)
        damas_15 = damas[['forecast_error_mw', 'forecast_error_pct', 'forecast_load_mw']].resample('15min').ffill()
        print(f"[+] DAMAS load: {len(damas_15)} periods")

    # DA prices
    path = REPO_ROOT / "features" / "DamasPrices" / "data" / "da_prices.csv"
    da_15 = None
    if path.exists():
        da = pd.read_csv(path, parse_dates=['datetime']).set_index('datetime').sort_index()
        keep = [c for c in ['price_eur_mwh', 'demand_mw', 'supply_mw',
                'net_flow_cz', 'net_flow_pl', 'net_flow_hu', 'net_import',
                'price_lag24', 'price_change_24h'] if c in da.columns]
        da = _dedup_index(da[keep])
        da_15 = da.resample('15min').ffill()
        print(f"[+] DA prices: {len(da_15)} periods")

    # Market spreads
    path = REPO_ROOT / "MarketPriceGap" / "data" / "processed" / "hourly_market_prices.csv"
    mkt_15 = None
    if path.exists():
        mkt = pd.read_csv(path, parse_dates=['timestamp_hour']).set_index('timestamp_hour').sort_index()
        keep = [c for c in ['idm_vwap', 'idm_volume_mwh', 'imb_settlement_price',
                'spread_da_idm', 'spread_idm_imb', 'spread_da_imb'] if c in mkt.columns]
        mkt = _dedup_index(mkt[keep])
        mkt_15 = mkt.resample('15min').ffill()
        print(f"[+] Market spreads: {len(mkt_15)} periods")

    # Weather: multi-city Slovakia (actuals) + Bardejov DA forecast
    weather_15 = None
    multi_path = REPO_ROOT / "data" / "Bardejov" / "Weather" / "slovakia_multi_city_weather.csv"
    da_path = REPO_ROOT / "data" / "Bardejov" / "Weather" / "bardejov_da_forecasts.csv"
    if multi_path.exists():
        mc = pd.read_csv(multi_path, parse_dates=['time']).set_index('time').sort_index()
        mc = _dedup_index(mc)
        # Keep national aggregates + Bratislava (least correlated) + Bardejov (existing)
        weather = mc[['temp_national_mean', 'temp_national_spread', 'temp_bratislava',
                       'wind_national_mean', 'radiation_national_mean',
                       'pressure_national_mean', 'temp_bardejov',
                       'cloud_bardejov']].copy()
        weather = weather.rename(columns={
            'temp_bardejov': 'temperature',  # backward compat
            'cloud_bardejov': 'cloudcover',
        })
        # DA temperature forecast for target hour (D-1 GFS, Bardejov only — safe)
        if da_path.exists():
            da_wx = pd.read_csv(da_path, parse_dates=['time']).set_index('time').sort_index()
            da_wx = _dedup_index(da_wx)
            for col in ['gfs_seamless_temp_da1', 'best_match_temp_da1']:
                if col in da_wx.columns:
                    weather = weather.join(da_wx[[col]], how='left')
                    weather = weather.rename(columns={col: 'temp_forecast_da'})
                    break
        weather_15 = weather.resample('15min').ffill()
        print(f"[+] Weather: {len(weather_15)} periods (5-city national + Bardejov DA forecast)")
    else:
        # Fallback to single-city Bardejov
        actual_path = REPO_ROOT / "data" / "Bardejov" / "Weather" / "bardejov_weather_actual.csv"
        if actual_path.exists():
            act = pd.read_csv(actual_path, parse_dates=['time']).set_index('time').sort_index()
            act = _dedup_index(act)
            weather = act[['temperature_2m', 'windspeed_10m', 'shortwave_radiation',
                            'cloudcover', 'surface_pressure']].copy()
            weather.columns = ['temperature', 'windspeed', 'radiation', 'cloudcover', 'pressure']
            if da_path.exists():
                da_wx = pd.read_csv(da_path, parse_dates=['time']).set_index('time').sort_index()
                da_wx = _dedup_index(da_wx)
                for col in ['gfs_seamless_temp_da1', 'best_match_temp_da1']:
                    if col in da_wx.columns:
                        weather = weather.join(da_wx[[col]], how='left')
                        weather = weather.rename(columns={col: 'temp_forecast_da'})
                        break
            weather_15 = weather.resample('15min').ffill()
            print(f"[+] Weather: {len(weather_15)} periods (Bardejov only, fallback)")

    # Load nowcast OOS (H+2 through H+5)
    nowcast_15 = None
    oos_dir = REPO_ROOT / "LoadAnalysis" / "nowcast_5h" / "tuning" / "oos_predictions"
    oos_path = oos_dir / "h2_oos_predictions.csv"
    if oos_path.exists():
        nc = pd.read_csv(oos_path, parse_dates=['datetime']).set_index('datetime').sort_index()
        nc = _dedup_index(nc)
        nowcast = nc[['predicted_error', 'actual_error']].copy()
        nowcast.columns = ['nowcast_pred_error', 'nowcast_actual_error']
        nowcast_15 = nowcast.resample('15min').ffill()
        print(f"[+] Load nowcast OOS H+2: {len(nowcast_15)} periods")

        # Load H+3, H+4, H+5 for multi-horizon features
        for h in [3, 4, 5]:
            h_path = oos_dir / f"h{h}_oos_predictions.csv"
            if h_path.exists():
                nc_h = pd.read_csv(h_path, parse_dates=['datetime']).set_index('datetime').sort_index()
                nc_h = _dedup_index(nc_h)
                nc_h_15 = nc_h[['predicted_error']].rename(
                    columns={'predicted_error': f'nowcast_h{h}_pred'}).resample('15min').ffill()
                nowcast_15 = nowcast_15.join(nc_h_15, how='left')
                print(f"[+] Load nowcast OOS H+{h}: {len(nc_h_15)} periods")

    # Imbalance labels
    path = REPO_ROOT / "data" / "master" / "master_imbalance_data.csv"
    imb = pd.read_csv(path, parse_dates=['datetime']).set_index('datetime').sort_index()
    imb = imb.rename(columns={
        'System Imbalance (MWh)': 'imbalance_mwh',
        'Imbalance Settlement Price (EUR/MWh)': 'imb_settle_price',
    })
    imb = imb[['imbalance_mwh', 'imb_settle_price']]
    print(f"[+] Imbalance: {len(imb)} periods")

    return {
        'reg': reg_15, 'load': load_15, 'prod': prod_15, 'xborder': xi_15,
        'solar': solar_15, 'damas': damas_15, 'da': da_15, 'mkt': mkt_15,
        'weather': weather_15, 'nowcast': nowcast_15, 'imb': imb,
    }


# ============================================================
# FEATURE ENGINEERING (parameterized by lead)
# ============================================================

def build_features(data, lead):
    """Build features for a given lead time. Returns (df, feature_cols)."""
    reg = data['reg']
    df = reg.join(data['load'], how='inner')
    for key in ['prod', 'xborder', 'solar', 'damas', 'da', 'mkt', 'weather', 'nowcast']:
        if data[key] is not None:
            df = df.join(data[key], how='left')
    df = df.join(data['imb'], how='inner')

    df['hour'] = df.index.hour
    df['qh'] = df.index.minute // 15
    df['hour_qh'] = df['hour'] * 4 + df['qh']
    df['dow'] = df.index.dayofweek
    train_mask = df.index <= TRAIN_END
    test_mask = df.index > TRAIN_END

    # SCADA shift = lead + 1: the 15-min period at T covers [T, T+15min)
    # and is only complete at T+15min. So at prediction time T-lead*15min,
    # the last COMPLETE period ended 1 period earlier.
    scada_shift = lead + 1
    # Hourly data shift = lead + 4 (1h publication delay on top of lead)
    hourly_shift = lead + 4

    features = {}

    # --- PROXY (lags + derived) ---
    for lag in range(scada_shift, scada_shift + 16):
        features[f'proxy_lag{lag}'] = df['proxy'].shift(lag)

    proxy_s = df['proxy'].shift(scada_shift)
    for w in [4, 8, 16, 32]:
        features[f'proxy_rmean{w}'] = proxy_s.rolling(w).mean()
    for w in [4, 8]:
        features[f'proxy_rstd{w}'] = proxy_s.rolling(w).std()
    features['proxy_rmin4'] = proxy_s.rolling(4).min()
    features['proxy_rmax4'] = proxy_s.rolling(4).max()
    features['proxy_range4'] = features['proxy_rmax4'] - features['proxy_rmin4']

    features['proxy_momentum'] = features[f'proxy_lag{scada_shift}'] - features[f'proxy_lag{scada_shift+1}']
    features['proxy_momentum4'] = features[f'proxy_lag{scada_shift}'] - features[f'proxy_lag{scada_shift+4}']
    features['proxy_acceleration'] = features['proxy_momentum'] - (
        features[f'proxy_lag{scada_shift+1}'] - features[f'proxy_lag{scada_shift+2}'])

    features['proxy_lag_pos'] = features[f'proxy_lag{scada_shift}'].clip(lower=0)
    features['proxy_lag_neg'] = features[f'proxy_lag{scada_shift}'].clip(upper=0)

    for w in [4, 8, 16]:
        pos_count = sum((df['proxy'].shift(scada_shift + i) > 0).astype(float) for i in range(w))
        features[f'proxy_pos_ratio_{w}'] = pos_count / w

    features['proxy_yesterday'] = df['proxy'].shift(96)
    features['proxy_yesterday_2'] = df['proxy'].shift(96 * 2)

    # Proxy deviation from hourly baseline
    proxy_train = proxy_s[train_mask]
    frozen_proxy_mean = proxy_train.groupby(df.loc[train_mask, 'hour_qh']).mean()
    proxy_bl = proxy_train.groupby(df.loc[train_mask, 'hour_qh']).expanding().mean()
    proxy_bl = proxy_bl.droplevel(0).sort_index()
    proxy_bl = proxy_bl[~proxy_bl.index.duplicated(keep='last')]
    proxy_baseline = proxy_bl.reindex(df.index)
    proxy_baseline[test_mask] = df.loc[test_mask, 'hour_qh'].map(frozen_proxy_mean).values
    features['proxy_dev_from_hour'] = proxy_s - proxy_baseline

    # Enhanced proxy
    features['proxy_ewm4'] = proxy_s.ewm(span=4).mean()
    features['proxy_range8'] = proxy_s.rolling(8).max() - proxy_s.rolling(8).min()
    features['proxy_zero_cross4'] = sum(
        (np.sign(df['proxy'].shift(scada_shift + i)) != np.sign(df['proxy'].shift(scada_shift + i + 1))).astype(float)
        for i in range(4))
    features['proxy_abs_rmean4'] = proxy_s.abs().rolling(4).mean()
    features['proxy_abs_rmean8'] = proxy_s.abs().rolling(8).mean()
    features['proxy_lag96_diff'] = features[f'proxy_lag{scada_shift}'] - df['proxy'].shift(96)

    # --- REGULATION ---
    reg_s = df['reg_mean'].shift(scada_shift)
    features['reg_rmean4'] = reg_s.rolling(4).mean()
    features['reg_rmean8'] = reg_s.rolling(8).mean()
    features['reg_rstd8'] = reg_s.rolling(8).std()
    features['reg_momentum'] = df['reg_mean'].shift(scada_shift) - df['reg_mean'].shift(scada_shift + 1)
    features['reg_vol_rmean4'] = df['reg_std'].shift(scada_shift).rolling(4).mean()

    # --- LOAD ---
    load_s = df['load_mean'].shift(scada_shift)
    features['load_rmean4'] = load_s.rolling(4).mean()
    features['load_rmean8'] = load_s.rolling(8).mean()
    features['load_rmean16'] = load_s.rolling(16).mean()
    features['load_momentum'] = df['load_mean'].shift(scada_shift) - df['load_mean'].shift(scada_shift + 4)

    load_train = load_s[train_mask]
    frozen_load_mean = load_train.groupby(df.loc[train_mask, 'hour_qh']).mean()
    load_bl = load_train.groupby(df.loc[train_mask, 'hour_qh']).expanding().mean()
    load_bl = load_bl.droplevel(0).sort_index()
    load_bl = load_bl[~load_bl.index.duplicated(keep='last')]
    load_baseline = load_bl.reindex(df.index)
    load_baseline[test_mask] = df.loc[test_mask, 'hour_qh'].map(frozen_load_mean).values
    features['load_deviation'] = load_s - load_baseline

    features['load_rstd4'] = load_s.rolling(4).std()
    load_max_s = df['load_max'].shift(scada_shift) if 'load_max' in df.columns else load_s
    load_min_s = df['load_min'].shift(scada_shift) if 'load_min' in df.columns else load_s
    features['load_ramp4'] = load_max_s.rolling(4).max() - load_min_s.rolling(4).min()
    features['load_yesterday'] = df['load_mean'].shift(96)

    # --- PRODUCTION ---
    if 'prod_mean' in df.columns:
        prod_s = df['prod_mean'].shift(scada_shift)
        features['prod_rmean4'] = prod_s.rolling(4).mean()
        features['prod_rmean8'] = prod_s.rolling(8).mean()
        features['prod_momentum'] = df['prod_mean'].shift(scada_shift) - df['prod_mean'].shift(scada_shift + 4)
        prod_train = prod_s[train_mask].dropna()
        if len(prod_train) > 96:
            frozen_prod = prod_train.groupby(df.loc[prod_train.index, 'hour_qh']).mean()
            features['prod_deviation'] = prod_s - df['hour_qh'].map(frozen_prod)
        else:
            features['prod_deviation'] = pd.Series(np.nan, index=df.index)
        features['prod_vol'] = (df['prod_std'].shift(scada_shift).rolling(4).mean()
                                if 'prod_std' in df.columns else pd.Series(np.nan, index=df.index))
    else:
        for f in ['prod_rmean4', 'prod_rmean8', 'prod_momentum', 'prod_deviation', 'prod_vol']:
            features[f] = pd.Series(np.nan, index=df.index)

    # --- EXPORT/IMPORT ---
    if 'xborder_mean' in df.columns:
        xb_s = df['xborder_mean'].shift(scada_shift)
        features['xborder_rmean4'] = xb_s.rolling(4).mean()
        features['xborder_rmean8'] = xb_s.rolling(8).mean()
        features['xborder_momentum'] = df['xborder_mean'].shift(scada_shift) - df['xborder_mean'].shift(scada_shift + 4)
        xb_train = xb_s[train_mask].dropna()
        if len(xb_train) > 96:
            frozen_xb = xb_train.groupby(df.loc[xb_train.index, 'hour_qh']).mean()
            features['xborder_deviation'] = xb_s - df['hour_qh'].map(frozen_xb)
        else:
            features['xborder_deviation'] = pd.Series(np.nan, index=df.index)
        features['xborder_vol'] = (df['xborder_std'].shift(scada_shift).rolling(4).mean()
                                   if 'xborder_std' in df.columns else pd.Series(np.nan, index=df.index))
    else:
        for f in ['xborder_rmean4', 'xborder_rmean8', 'xborder_momentum', 'xborder_deviation', 'xborder_vol']:
            features[f] = pd.Series(np.nan, index=df.index)

    # --- SOLAR (hourly, shift = lead + 4) ---
    if 'solar_surprise_mw' in df.columns:
        solar_s = df['solar_surprise_mw'].shift(hourly_shift)
        features['solar_surprise_lag'] = solar_s
        features['solar_surprise_rmean4'] = solar_s.rolling(4).mean()
    else:
        features['solar_surprise_lag'] = 0
        features['solar_surprise_rmean4'] = 0

    # --- DAMAS FE (hourly, shift = lead + 4) ---
    if 'forecast_error_mw' in df.columns:
        fe_s = df['forecast_error_mw'].shift(hourly_shift)
        features['damas_fe'] = fe_s
        features['damas_fe_abs'] = fe_s.abs()
        features['damas_fe_rmean4'] = fe_s.rolling(4).mean()
        features['damas_fe_rmean24'] = fe_s.rolling(96).mean()
        features['damas_forecast_load'] = df['forecast_load_mw']  # D-1, no shift
    else:
        for f in ['damas_fe', 'damas_fe_abs', 'damas_fe_rmean4', 'damas_fe_rmean24', 'damas_forecast_load']:
            features[f] = pd.Series(np.nan, index=df.index)

    # --- DA PRICES (D-1, no shift) ---
    if 'price_eur_mwh' in df.columns:
        features['da_price'] = df['price_eur_mwh']
        features['da_price_change24h'] = df.get('price_change_24h', pd.Series(np.nan, index=df.index))
        features['da_demand'] = df.get('demand_mw', pd.Series(np.nan, index=df.index))
        features['da_supply'] = df.get('supply_mw', pd.Series(np.nan, index=df.index))
        features['da_net_import'] = df.get('net_import', pd.Series(np.nan, index=df.index))
        features['da_flow_cz'] = df.get('net_flow_cz', pd.Series(np.nan, index=df.index))
        features['da_flow_hu'] = df.get('net_flow_hu', pd.Series(np.nan, index=df.index))
    else:
        for f in ['da_price', 'da_price_change24h', 'da_demand', 'da_supply',
                   'da_net_import', 'da_flow_cz', 'da_flow_hu']:
            features[f] = pd.Series(np.nan, index=df.index)

    # --- MARKET SPREADS (hourly, shift = lead + 4) ---
    if 'idm_vwap' in df.columns:
        features['idm_vwap_lag'] = df['idm_vwap'].shift(hourly_shift)
        features['idm_volume_lag'] = df.get('idm_volume_mwh', pd.Series(np.nan, index=df.index)).shift(hourly_shift)
        features['spread_da_idm_lag'] = df.get('spread_da_idm', pd.Series(np.nan, index=df.index)).shift(hourly_shift)
    else:
        for f in ['idm_vwap_lag', 'idm_volume_lag', 'spread_da_idm_lag']:
            features[f] = pd.Series(np.nan, index=df.index)

    if 'imb_settle_price' in df.columns:
        features['imb_price_lag'] = df['imb_settle_price'].shift(hourly_shift)
        features['imb_price_rmean4'] = df['imb_settle_price'].shift(hourly_shift).rolling(4).mean()
        features['spread_da_imb_lag'] = (df['price_eur_mwh'] - df['imb_settle_price'].shift(hourly_shift)
                                         if 'price_eur_mwh' in df.columns else pd.Series(np.nan, index=df.index))
    else:
        for f in ['imb_price_lag', 'imb_price_rmean4', 'spread_da_imb_lag']:
            features[f] = pd.Series(np.nan, index=df.index)

    # --- WEATHER ---
    # Actuals: shifted by hourly_shift (lead+4, ~3h). These are observations,
    #          available after the hour ends with ~1h publication delay.
    # DA forecast: no shift — D-1 GFS forecast for the target hour. Known D-1.
    #              Only available for Bardejov (we have historical D-1 forecasts).
    #              NOT using intraday forecast updates — can't verify timing in backtest.
    if 'temperature' in df.columns:
        # --- Bardejov actuals (backward-compatible) ---
        temp_s = df['temperature'].shift(hourly_shift)
        features['temp'] = temp_s
        features['temp_change6h'] = temp_s - df['temperature'].shift(hourly_shift + 24)
        features['temp_rmean24h'] = temp_s.rolling(96).mean()
        features['temp_deviation'] = temp_s - temp_s.rolling(96 * 7, min_periods=96).mean()
        features['cloudcover'] = df.get('cloudcover', pd.Series(np.nan, index=df.index)).shift(hourly_shift)

        # --- DA forecast for target hour (Bardejov D-1 GFS, no shift — safe) ---
        if 'temp_forecast_da' in df.columns:
            features['temp_forecast_da'] = df['temp_forecast_da']
            features['temp_surprise_lag'] = (df['temperature'] - df['temp_forecast_da']).shift(hourly_shift)
        else:
            features['temp_forecast_da'] = pd.Series(np.nan, index=df.index)
            features['temp_surprise_lag'] = pd.Series(np.nan, index=df.index)

        # --- National aggregates (5-city actuals, shifted — safe) ---
        if 'temp_national_mean' in df.columns:
            nat_temp_s = df['temp_national_mean'].shift(hourly_shift)
            features['temp_national'] = nat_temp_s
            features['temp_national_change6h'] = nat_temp_s - df['temp_national_mean'].shift(hourly_shift + 24)
            features['temp_national_deviation'] = nat_temp_s - nat_temp_s.rolling(96 * 7, min_periods=96).mean()
            features['temp_national_spread'] = df['temp_national_spread'].shift(hourly_shift)
        else:
            for f in ['temp_national', 'temp_national_change6h', 'temp_national_deviation', 'temp_national_spread']:
                features[f] = pd.Series(np.nan, index=df.index)

        # --- Bratislava (least correlated with east, biggest load center) ---
        if 'temp_bratislava' in df.columns:
            features['temp_bratislava'] = df['temp_bratislava'].shift(hourly_shift)
        else:
            features['temp_bratislava'] = pd.Series(np.nan, index=df.index)

        # --- National wind, radiation, pressure (actuals, shifted) ---
        if 'wind_national_mean' in df.columns:
            features['wind_national'] = df['wind_national_mean'].shift(hourly_shift)
        else:
            features['wind_national'] = df.get('windspeed', pd.Series(np.nan, index=df.index)).shift(hourly_shift)

        if 'radiation_national_mean' in df.columns:
            features['radiation_national'] = df['radiation_national_mean'].shift(hourly_shift)
        else:
            features['radiation_national'] = df.get('radiation', pd.Series(np.nan, index=df.index)).shift(hourly_shift)

        if 'pressure_national_mean' in df.columns:
            pres_s = df['pressure_national_mean'].shift(hourly_shift)
            features['pressure_change6h'] = pres_s - df['pressure_national_mean'].shift(hourly_shift + 24)
        elif 'pressure' in df.columns:
            features['pressure_change6h'] = df['pressure'].shift(hourly_shift) - df['pressure'].shift(hourly_shift + 24)
        else:
            features['pressure_change6h'] = pd.Series(np.nan, index=df.index)
    else:
        for f in ['temp', 'temp_change6h', 'temp_rmean24h', 'temp_deviation',
                   'temp_forecast_da', 'temp_surprise_lag', 'cloudcover',
                   'temp_national', 'temp_national_change6h', 'temp_national_deviation',
                   'temp_national_spread', 'temp_bratislava',
                   'wind_national', 'radiation_national', 'pressure_change6h']:
            features[f] = pd.Series(np.nan, index=df.index)

    # --- LOAD NOWCAST (multi-horizon OOS) ---
    # OOS files are indexed by PREDICTION TIME. H+X prediction made at T targets T+Xh.
    # To get the prediction targeting delivery hour T, use the one made at T-Xh:
    #   shift = X * 4 (hours to 15-min periods)
    # All horizons target the SAME delivery hour from progressively earlier vantage points.
    # Momentum (H+2 - H+3) = how the forecast evolves as delivery approaches.
    if 'nowcast_pred_error' in df.columns:
        # H+2: shift by 2h = 8 periods (= lead for lead=8, staler for shorter leads)
        nowcast_h2 = df['nowcast_pred_error'].shift(2 * 4)
        features['nowcast_pred_error'] = nowcast_h2
        features['nowcast_pred_error_abs'] = nowcast_h2.abs()
        features['nowcast_pred_rmean4'] = nowcast_h2.rolling(4).mean()

        # H+3, H+4, H+5: earlier predictions of the same delivery hour
        prev_h = nowcast_h2
        for h in [3, 4, 5]:
            col = f'nowcast_h{h}_pred'
            if col in df.columns:
                nowcast_hx = df[col].shift(h * 4)
                features[f'nowcast_h{h}'] = nowcast_hx
                # Momentum: how forecast changes between adjacent horizons
                # H+2 - H+3 = change from 3h-ago prediction to 2h-ago prediction
                features[f'nowcast_momentum_h{h-1}h{h}'] = prev_h - nowcast_hx
                prev_h = nowcast_hx

        # Overall nowcast trend: H+2 vs H+5 (most recent vs earliest)
        if 'nowcast_h5_pred' in df.columns:
            nowcast_h5 = df['nowcast_h5_pred'].shift(5 * 4)
            features['nowcast_trend_h2_h5'] = nowcast_h2 - nowcast_h5
            features['nowcast_convergence'] = nowcast_h2.abs() - nowcast_h5.abs()
    else:
        for f in ['nowcast_pred_error', 'nowcast_pred_error_abs', 'nowcast_pred_rmean4']:
            features[f] = pd.Series(np.nan, index=df.index)

    # --- TIME ---
    features['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    features['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    features['qh_in_hour'] = df['qh']
    features['is_weekend'] = (df['dow'] >= 5).astype(int)
    features['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    features['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    features['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    features['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
    features['is_peak'] = ((df['hour'] >= 7) & (df['hour'] <= 22)).astype(int)

    # --- ASSEMBLE ---
    feat_df = pd.DataFrame(features, index=df.index)
    feature_cols = list(feat_df.columns)
    feat_df['target'] = df['imbalance_mwh']
    # Preserve settlement price for spread target computation downstream
    if 'imb_settle_price' in df.columns:
        feat_df['imb_settle_price'] = df['imb_settle_price']

    core_cols = [f'proxy_lag{scada_shift}', 'proxy_rmean4', 'load_rmean4', 'hour_sin', 'target']
    valid = feat_df.dropna(subset=core_cols)

    return valid, feature_cols


# ============================================================
# TRAINING & EVALUATION
# ============================================================

def train_lead(data, lead):
    """Train quantile models for one lead time."""
    print(f"\n{'='*60}")
    print(f"LEAD {lead} ({lead * 15} min = {lead * 15 / 60:.2g}h ahead)")
    print(f"{'='*60}")

    df, feature_cols = build_features(data, lead)
    print(f"[+] Features: {len(feature_cols)}, Valid samples: {len(df)}")

    train = df[df.index <= TRAIN_END]
    test = df[df.index >= TEST_START]
    print(f"    Train: {len(train)}, Test: {len(test)}")

    X_train, y_train = train[feature_cols].values, train['target'].values
    X_test, y_test = test[feature_cols].values, test['target'].values

    models = {}
    predictions = {}

    for q in QUANTILES:
        model = lgb.LGBMRegressor(
            objective='quantile', alpha=q, learning_rate=0.05,
            num_leaves=63, min_child_samples=50, subsample=0.8,
            colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
            n_estimators=600, verbose=-1,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                  callbacks=[lgb.log_evaluation(0)])
        predictions[f'q{int(q*100)}'] = model.predict(X_test)
        models[q] = model
        joblib.dump(model, MODEL_DIR / f"imb_lead{lead}_q{int(q*100)}.joblib")

    pred_df = test[['target']].copy()
    for q_name, pred in predictions.items():
        pred_df[q_name] = pred
    pred_df['pred_median'] = pred_df['q50']
    pred_df['pred_direction'] = np.sign(pred_df['pred_median'])
    pred_df['actual_direction'] = np.sign(pred_df['target'])

    nonzero = pred_df['target'].abs() > 0.1
    dir_acc = (pred_df.loc[nonzero, 'pred_direction'] == pred_df.loc[nonzero, 'actual_direction']).mean()

    high_conf = pred_df['pred_median'].abs() > 5
    hc_acc = (pred_df.loc[high_conf & nonzero, 'pred_direction'] ==
              pred_df.loc[high_conf & nonzero, 'actual_direction']).mean() if high_conf.sum() > 0 else 0

    print(f"\n  Direction accuracy: {dir_acc:.1%}")
    print(f"  High conf (|pred|>5): {hc_acc:.1%} ({(high_conf & nonzero).sum()} periods)")

    pred_df.to_csv(DATA_DIR / f"predictions_lead{lead}.csv")
    return pred_df, dir_acc, hc_acc


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("MULTI-LEAD IMBALANCE PREDICTOR (Leads 4-8)")
    print("=" * 70)

    data = load_all_data()

    results = {}
    for lead in LEADS:
        pred_df, dir_acc, hc_acc = train_lead(data, lead)
        results[lead] = {'pred_df': pred_df, 'dir_acc': dir_acc, 'hc_acc': hc_acc}

    # Summary
    print("\n" + "=" * 70)
    print("MULTI-LEAD SUMMARY")
    print("=" * 70)
    print(f"\n  {'Lead':>6s}  {'Time':>6s}  {'Dir Acc':>8s}  {'HC Acc':>8s}")
    print("  " + "-" * 35)
    for lead in LEADS:
        r = results[lead]
        print(f"  Lead {lead}  {lead*15:>4d}m   {r['dir_acc']:>7.1%}   {r['hc_acc']:>7.1%}")

    print(f"\n[+] All predictions saved to {DATA_DIR}")
    print(f"[+] All models saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
