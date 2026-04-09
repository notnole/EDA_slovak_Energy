"""
Speculative Spread Trading Strategies for Slovak Electricity Market
===================================================================
Three ML-based spread strategies with walk-forward validation:
  A) DA-IDM spread prediction (LightGBM, D-1 features)
  B) DA-Imbalance spread with risk management
  C) IDM-Imbalance with morning signal (intraday)

Enhanced with:
  - Renewable generation + cross-border flow features (ProductionPerType.csv)
  - Half-Kelly position sizing
  - Hour-filtering for high-variance periods
  - Multi-spread portfolio combination

Usage:
  python spread_strategies.py
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')

BASE = Path(r'C:\Users\20254757\pycharmprojects\ipesoft_eda_data')
OUT = BASE / 'MarketPriceGap' / 'strategies' / 'speculative'
OUT.mkdir(parents=True, exist_ok=True)

# High-variance hours per spread (from empirical analysis)
HIGH_VAR_HOURS_DA_IDM = [13, 14, 17, 19, 20]  # std > 30
HIGH_VAR_HOURS_DA_IMB = [6, 9, 10, 12, 15]     # std > 100
HIGH_VAR_HOURS_IDM_IMB = [7, 10, 12, 14, 15]   # std > 85


# ============================================================
# 1. DATA LOADING
# ============================================================

def parse_production_data():
    """Parse ProductionPerType.csv for renewable/nuclear/gas features."""
    filepath = BASE / 'RawData' / 'ProductionPerType.csv'
    if not filepath.exists():
        print("[-] ProductionPerType.csv not found")
        return None

    print("[*] Parsing production data (nuclear, solar, gas, hydro)...")
    col_names = ['Biomass', 'FosilOil', 'HardCoal', 'HydroPump',
                 'HydroPumpConsumption', 'HydroReservoir', 'HydroRunRiver',
                 'Lignite', 'NatGas', 'Nuclear', 'Other', 'OtherRenewable',
                 'SolarActual', 'SolarForecast']

    dt_pat = re.compile(r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}')
    records = []

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            matches = list(dt_pat.finditer(line))
            if not matches:
                continue
            try:
                dt = pd.to_datetime(matches[0].group(), format='%m/%d/%Y %H:%M:%S')
            except Exception:
                continue

            row = {'timestamp_hour': dt}
            for i, match in enumerate(matches):
                if i >= len(col_names):
                    break
                start = match.end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(line)
                seg = line[start:end]
                parts = [p.strip() for p in seg.split(',') if p.strip()]
                if len(parts) >= 2 and parts[1] != '(invalid)':
                    try:
                        if len(parts) >= 3 and parts[2].isdigit() and len(parts[2]) <= 2:
                            row[col_names[i]] = float(f"{parts[1]}.{parts[2]}")
                        else:
                            row[col_names[i]] = float(parts[1])
                    except Exception:
                        row[col_names[i]] = np.nan
                else:
                    row[col_names[i]] = np.nan
            records.append(row)

    prod = pd.DataFrame(records)
    # Derived: total renewable, total thermal
    prod['renewable_total'] = (prod.get('SolarActual', 0).fillna(0) +
                                prod.get('HydroRunRiver', 0).fillna(0) +
                                prod.get('HydroReservoir', 0).fillna(0) +
                                prod.get('OtherRenewable', 0).fillna(0) +
                                prod.get('Biomass', 0).fillna(0))
    prod['thermal_total'] = (prod.get('NatGas', 0).fillna(0) +
                              prod.get('Lignite', 0).fillna(0) +
                              prod.get('HardCoal', 0).fillna(0))
    # Solar forecast error (when available)
    if 'SolarForecast' in prod.columns and 'SolarActual' in prod.columns:
        prod['solar_fcst_error'] = prod['SolarForecast'] - prod['SolarActual']

    valid = prod['Nuclear'].notna().sum()
    print(f"[+] Production: {len(prod)} rows, {valid} valid nuclear obs")
    return prod


def load_data():
    """Load and merge all data sources including production."""
    print("[*] Loading data...")

    # Main market prices
    mkt = pd.read_csv(BASE / 'MarketPriceGap' / 'data' / 'processed' / 'hourly_market_prices.csv')
    mkt['timestamp_hour'] = pd.to_datetime(mkt['timestamp_hour'])
    mkt['date'] = pd.to_datetime(mkt['date'])
    if mkt['is_weekend'].dtype == object:
        mkt['is_weekend'] = mkt['is_weekend'].str.lower().isin(['true', '1'])
    else:
        mkt['is_weekend'] = mkt['is_weekend'].astype(bool)
    print(f"[+] Market: {len(mkt)} rows, {mkt['date'].min().date()} to {mkt['date'].max().date()}")

    # Load forecasts
    load = pd.read_csv(BASE / 'features' / 'DamasLoad' / 'load_data.csv')
    load['timestamp_hour'] = pd.to_datetime(load['datetime'])
    load = load[['timestamp_hour', 'actual_load_mw', 'forecast_load_mw', 'forecast_error_mw']]
    print(f"[+] Load: {len(load)} rows")

    # DA price forecasts (15min -> hourly)
    da_fcst_path = BASE / 'data' / 'clean' / 'market' / 'da_forecasts' / 'da_price_forecasts_15min.csv'
    da_fcst = None
    if da_fcst_path.exists():
        df_fcst = pd.read_csv(da_fcst_path)
        df_fcst['datetime'] = pd.to_datetime(df_fcst['datetime'])
        for col in ['sk_merged_spot', 'sk_forecast1_spot', 'sk_forecast2_spot']:
            if col in df_fcst.columns:
                df_fcst['da_forecast'] = df_fcst[col]
                break
        if 'da_forecast' in df_fcst.columns:
            df_fcst['timestamp_hour'] = df_fcst['datetime'].dt.floor('h')
            da_fcst = df_fcst.groupby('timestamp_hour')['da_forecast'].mean().reset_index()
            valid = da_fcst['da_forecast'].notna().sum()
            print(f"[+] DA forecasts: {valid} valid hourly values")

    # Production data (renewable, nuclear, gas, hydro)
    prod = parse_production_data()

    # Merge
    df = mkt.merge(load, on='timestamp_hour', how='left')
    if da_fcst is not None:
        df = df.merge(da_fcst, on='timestamp_hour', how='left')
    else:
        df['da_forecast'] = np.nan

    if prod is not None and len(prod) > 0:
        prod_cols = ['timestamp_hour', 'Nuclear', 'NatGas', 'SolarActual', 'SolarForecast',
                     'HydroPump', 'HydroRunRiver', 'renewable_total', 'thermal_total',
                     'solar_fcst_error']
        prod_available = [c for c in prod_cols if c in prod.columns]
        df = df.merge(prod[prod_available], on='timestamp_hour', how='left')
    else:
        for c in ['Nuclear', 'NatGas', 'SolarActual', 'HydroPump', 'HydroRunRiver',
                   'renewable_total', 'thermal_total']:
            df[c] = np.nan

    df = df.sort_values(['date', 'hour']).reset_index(drop=True)
    print(f"[+] Merged: {len(df)} rows, {df.shape[1]} columns")
    return df


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def build_features(df):
    """Build D-1 available features for all strategies."""
    print("[*] Building features...")

    # Calendar features
    df['dow'] = df['timestamp_hour'].dt.dayofweek
    df['month_num'] = df['timestamp_hour'].dt.month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
    df['is_weekend_int'] = df['is_weekend'].astype(int)

    # --- Daily aggregates from YESTERDAY ---
    agg_dict = {
        'da_price': [('da_mean', 'mean'), ('da_max', 'max'), ('da_min', 'min'),
                     ('da_std', 'std')],
        'idm_vwap': [('idm_mean', 'mean')],
        'spread_da_idm': [('spread_da_idm_mean', 'mean')],
        'spread_da_imb': [('spread_da_imb_mean', 'mean')],
        'spread_idm_imb': [('spread_idm_imb_mean', 'mean')],
        'forecast_load_mw': [('load_fcst_mean', 'mean')],
        'actual_load_mw': [('load_actual_mean', 'mean')],
        'forecast_error_mw': [('load_error_mean', 'mean')],
        'imb_price': [('imb_mean', 'mean'), ('imb_std', 'std')],
        'imbalance_mwh': [('imbalance_total', 'sum')],
        'net_import': [('net_import_mean', 'mean')],
    }

    # Add production columns if available
    for pcol in ['Nuclear', 'NatGas', 'SolarActual', 'HydroPump', 'renewable_total', 'thermal_total']:
        if pcol in df.columns:
            agg_dict[pcol] = [(f'{pcol.lower()}_mean', 'mean')]

    daily_aggs = {}
    for col, agg_list in agg_dict.items():
        if col not in df.columns:
            continue
        for new_name, agg_func in agg_list:
            if new_name not in daily_aggs:
                daily_aggs[new_name] = (col, agg_func)

    daily = df.groupby('date').agg(**daily_aggs).reset_index()

    # DA price range
    if 'da_max' in daily.columns and 'da_min' in daily.columns:
        daily['da_range'] = daily['da_max'] - daily['da_min']

    daily = daily.sort_values('date').reset_index(drop=True)

    # Yesterday shifts
    skip_cols = {'date'}
    for col in daily.columns:
        if col not in skip_cols:
            daily[f'{col}_yest'] = daily[col].shift(1)

    # 7-day rolling
    roll_cols = [c for c in ['da_mean', 'idm_mean', 'spread_da_idm_mean', 'spread_da_imb_mean',
                              'spread_idm_imb_mean', 'load_actual_mean', 'net_import_mean',
                              'nuclear_mean', 'solaractual_mean', 'renewable_total_mean']
                 if c in daily.columns]
    for col in roll_cols:
        daily[f'{col}_7d'] = daily[col].shift(1).rolling(7, min_periods=3).mean()

    # 3-day rolling
    for col in ['spread_da_idm_mean', 'spread_da_imb_mean', 'spread_idm_imb_mean']:
        if col in daily.columns:
            daily[f'{col}_3d'] = daily[col].shift(1).rolling(3, min_periods=2).mean()

    # Load forecast deviation from recent norm
    if 'load_fcst_mean' in daily.columns and 'load_actual_mean_7d' in daily.columns:
        daily['load_fcst_dev'] = daily['load_fcst_mean'] - daily['load_actual_mean_7d']

    # Spread momentum (yesterday - 7d avg)
    for s in ['spread_da_idm_mean', 'spread_da_imb_mean', 'spread_idm_imb_mean']:
        if f'{s}_yest' in daily.columns and f'{s}_7d' in daily.columns:
            daily[f'{s}_momentum'] = daily[f'{s}_yest'] - daily[f'{s}_7d']

    # Merge daily features back - avoid duplicating columns already in df
    merge_cols = [c for c in daily.columns if c not in df.columns or c == 'date']
    df = df.merge(daily[merge_cols], on='date', how='left')

    # Same-hour yesterday values
    df = df.sort_values(['hour', 'date']).reset_index(drop=True)
    for col in ['da_price', 'idm_vwap', 'spread_da_idm', 'spread_da_imb',
                'spread_idm_imb', 'imb_price', 'net_import']:
        if col in df.columns:
            df[f'{col}_yest_h'] = df.groupby('hour')[col].shift(1)
            df[f'{col}_7d_h'] = df.groupby('hour')[col].shift(1).rolling(7, min_periods=3).mean()

    # Production features: same-hour yesterday
    for col in ['Nuclear', 'NatGas', 'SolarActual', 'renewable_total']:
        if col in df.columns:
            df[f'{col}_yest_h'] = df.groupby('hour')[col].shift(1)

    # Same-hour 7d rolling std (volatility feature)
    for col in ['spread_da_idm', 'spread_da_imb', 'spread_idm_imb']:
        if col in df.columns:
            df[f'{col}_vol_7d_h'] = df.groupby('hour')[col].shift(1).rolling(7, min_periods=3).std()

    df = df.sort_values(['date', 'hour']).reset_index(drop=True)

    # DA forecast features (available D-1)
    if 'da_forecast' in df.columns:
        df['da_fcst_minus_yest'] = df['da_forecast'] - df.get('da_price_yest_h', np.nan)
        df['da_fcst_minus_7d'] = df['da_forecast'] - df.get('da_price_7d_h', np.nan)

    print(f"[+] Features: {df.shape[1]} columns")
    return df


# ============================================================
# HALF-KELLY POSITION SIZING
# ============================================================

def half_kelly(win_rate, avg_win, avg_loss):
    """Calculate Half-Kelly fraction.

    Kelly f* = (p*b - q) / b  where p=win_rate, q=1-p, b=avg_win/avg_loss
    Half-Kelly = f*/2 for safety.
    Returns fraction in [0, MAX_POSITION].
    """
    MAX_POSITION = 1.5
    MIN_POSITION = 0.1

    if avg_loss == 0 or win_rate <= 0:
        return MIN_POSITION

    b = abs(avg_win / avg_loss)
    q = 1 - win_rate
    kelly = (win_rate * b - q) / b
    hk = kelly / 2

    return np.clip(hk, MIN_POSITION, MAX_POSITION)


def compute_rolling_kelly(traded_pnl, window=200):
    """Compute rolling Half-Kelly from recent trade P&L."""
    if len(traded_pnl) < 50:
        return 0.5  # Default conservative

    recent = traded_pnl[-window:]
    wins = recent[recent > 0]
    losses = recent[recent <= 0]

    if len(wins) == 0 or len(losses) == 0:
        return 0.5

    wr = len(wins) / len(recent)
    avg_w = wins.mean()
    avg_l = losses.mean()

    return half_kelly(wr, avg_w, avg_l)


# ============================================================
# 3. STRATEGY A: DA-IDM Spread (Enhanced)
# ============================================================

FEAT_A = [
    'hour', 'hour_sin', 'hour_cos', 'dow', 'dow_sin', 'dow_cos',
    'month_sin', 'month_cos', 'is_weekend_int',
    'da_mean_yest', 'da_std_yest', 'da_range_yest',
    'idm_mean_yest', 'spread_da_idm_mean_yest',
    'spread_da_idm_mean_7d', 'spread_da_idm_mean_3d',
    'spread_da_idm_mean_momentum',
    'load_fcst_dev', 'load_error_mean_yest',
    'da_price_yest_h', 'idm_vwap_yest_h', 'spread_da_idm_yest_h',
    'da_price_7d_h', 'spread_da_idm_7d_h', 'spread_da_idm_vol_7d_h',
    'load_fcst_mean',
    # New: production & flow features
    'net_import_yest_h', 'net_import_mean_yest', 'net_import_mean_7d',
    'nuclear_mean_yest', 'solaractual_mean_yest', 'renewable_total_mean_yest',
    'Nuclear_yest_h', 'NatGas_yest_h', 'SolarActual_yest_h', 'renewable_total_yest_h',
]


def run_strategy_a(df):
    """Strategy A: DA-IDM Spread (LightGBM walk-forward, hour-filtered, Half-Kelly)."""
    print("\n" + "=" * 60)
    print("  STRATEGY A: DA-IDM Spread (Enhanced)")
    print("=" * 60)

    # Filter: need both DA and IDM
    mask = df['spread_da_idm'].notna()
    data = df[mask].copy()
    if len(data) == 0:
        print("[!] No DA-IDM spread data available")
        return pd.DataFrame()

    # Target
    data['target_dir'] = (data['spread_da_idm'] > 0).astype(int)
    data['target_mag'] = data['spread_da_idm']

    # Available features
    avail_feats = [f for f in FEAT_A if f in data.columns]
    print(f"[*] Using {len(avail_feats)} features")

    # Walk-forward by month
    data['ym'] = data['date'].dt.to_period('M')
    months = sorted(data['ym'].unique())
    min_train_months = 3

    if len(months) <= min_train_months:
        print(f"[!] Only {len(months)} months, need >{min_train_months}")
        return pd.DataFrame()

    results = []
    recent_pnls = []

    for i in range(min_train_months, len(months)):
        test_month = months[i]
        train_months = months[:i]

        train = data[data['ym'].isin(train_months)]
        test = data[data['ym'] == test_month]

        if len(test) == 0:
            continue

        X_train = train[avail_feats].fillna(0)
        y_train_dir = train['target_dir']
        y_train_mag = train['target_mag']
        X_test = test[avail_feats].fillna(0)

        # Direction classifier
        clf = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.03,
            num_leaves=31, min_child_samples=30,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.2, reg_lambda=0.2,
            verbose=-1, n_jobs=-1
        )
        clf.fit(X_train, y_train_dir)
        proba = clf.predict_proba(X_test)[:, 1]

        # Magnitude regressor
        reg = lgb.LGBMRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.03,
            num_leaves=31, min_child_samples=30,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.2, reg_lambda=0.2,
            verbose=-1, n_jobs=-1
        )
        reg.fit(X_train, y_train_mag)
        pred_mag = reg.predict(X_test)

        # Rolling Kelly from recent performance
        kelly_frac = compute_rolling_kelly(np.array(recent_pnls))

        for idx, row in test.iterrows():
            j = test.index.get_loc(idx)
            p = proba[j]
            m = pred_mag[j]
            actual = row['spread_da_idm']
            h = row['hour']

            confidence = abs(p - 0.5) * 2
            signal = 1 if p > 0.5 else -1

            # Hour filter: focus on high-variance hours + weekends
            is_high_var = h in HIGH_VAR_HOURS_DA_IDM
            is_wknd = bool(row.get('is_weekend', False))

            # Trade when confident AND (high-var hour OR weekend OR large predicted magnitude)
            trade = 0
            if confidence > 0.15 and (is_high_var or is_wknd or abs(m) > 15):
                trade = 1

            # Half-Kelly position sizing scaled by confidence
            position = kelly_frac * (0.5 + 0.5 * confidence) if trade else 0
            position = min(position, 1.5)

            pnl = signal * actual * position if trade else 0

            if trade:
                recent_pnls.append(pnl)

            results.append({
                'timestamp_hour': row['timestamp_hour'],
                'date': row['date'],
                'hour': h,
                'spread_actual': actual,
                'pred_direction': int(p > 0.5),
                'pred_magnitude': m,
                'probability': p,
                'confidence': confidence,
                'position_size': position,
                'signal': signal,
                'trade': trade,
                'pnl': pnl,
                'correct': int((signal > 0) == (actual > 0)) if trade else np.nan,
                'strategy': 'A_DA_IDM',
                'month': str(test_month),
            })

    res = pd.DataFrame(results)
    if len(res) == 0:
        print("[!] No results generated")
        return res

    traded = res[res['trade'] == 1]
    print(f"\n[+] Strategy A Results:")
    print(f"    Period: {res['date'].min().date()} to {res['date'].max().date()}")
    print(f"    Total hours: {len(res)}, Traded: {len(traded)} ({len(traded)/len(res)*100:.0f}%)")
    if len(traded) > 0:
        print(f"    Total P&L: {traded['pnl'].sum():,.0f} EUR")
        print(f"    Accuracy: {traded['correct'].mean():.1%}")
        print(f"    Avg P&L/trade: {traded['pnl'].mean():.2f} EUR")
        std_t = traded['pnl'].std()
        print(f"    Sharpe (ann): {traded['pnl'].mean() / std_t * np.sqrt(252*24):.2f}" if std_t > 0 else "    Sharpe: N/A")

        # Feature importance
        feat_imp = pd.Series(clf.feature_importances_, index=avail_feats).sort_values(ascending=False)
        print(f"    Top features: {list(feat_imp.head(7).index)}")

    return res


# ============================================================
# 4. STRATEGY B: DA-Imbalance Spread (Enhanced)
# ============================================================

FEAT_B = [
    'hour', 'hour_sin', 'hour_cos', 'dow', 'dow_sin', 'dow_cos',
    'month_sin', 'month_cos', 'is_weekend_int',
    'da_mean_yest', 'da_std_yest', 'da_range_yest',
    'spread_da_imb_mean_yest', 'spread_da_imb_mean_7d', 'spread_da_imb_mean_3d',
    'spread_da_imb_mean_momentum',
    'imb_mean_yest', 'imb_std_yest', 'imbalance_total_yest',
    'load_fcst_dev', 'load_error_mean_yest',
    'da_price_yest_h', 'imb_price_yest_h', 'spread_da_imb_yest_h',
    'spread_da_imb_7d_h', 'spread_da_imb_vol_7d_h',
    'load_fcst_mean',
    # New: production & flow features
    'net_import_yest_h', 'net_import_mean_yest',
    'nuclear_mean_yest', 'solaractual_mean_yest', 'renewable_total_mean_yest',
    'Nuclear_yest_h', 'NatGas_yest_h', 'SolarActual_yest_h', 'renewable_total_yest_h',
]


def run_strategy_b(df):
    """Strategy B: DA-Imbalance Spread with Half-Kelly + risk management."""
    print("\n" + "=" * 60)
    print("  STRATEGY B: DA-Imbalance Spread (Enhanced)")
    print("=" * 60)

    mask = df['spread_da_imb'].notna()
    data = df[mask].copy()
    if len(data) == 0:
        print("[!] No DA-Imbalance spread data available")
        return pd.DataFrame()

    data['target_dir'] = (data['spread_da_imb'] > 0).astype(int)

    avail_feats = [f for f in FEAT_B if f in data.columns]
    print(f"[*] Using {len(avail_feats)} features")

    data['ym'] = data['date'].dt.to_period('M')
    months = sorted(data['ym'].unique())
    min_train_months = 3

    if len(months) <= min_train_months:
        print(f"[!] Only {len(months)} months, need >{min_train_months}")
        return pd.DataFrame()

    results = []
    recent_pnls = []

    for i in range(min_train_months, len(months)):
        test_month = months[i]
        train_months = months[:i]

        train = data[data['ym'].isin(train_months)]
        test = data[data['ym'] == test_month]

        if len(test) == 0:
            continue

        X_train = train[avail_feats].fillna(0)
        y_train_dir = train['target_dir']
        X_test = test[avail_feats].fillna(0)

        clf = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.03,
            num_leaves=31, min_child_samples=30,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.2, reg_lambda=0.2,
            verbose=-1, n_jobs=-1
        )
        clf.fit(X_train, y_train_dir)
        proba = clf.predict_proba(X_test)[:, 1]

        kelly_frac = compute_rolling_kelly(np.array(recent_pnls))

        for idx, row in test.iterrows():
            j = test.index.get_loc(idx)
            p = proba[j]
            actual = row['spread_da_imb']
            h = row['hour']

            confidence = abs(p - 0.5) * 2
            signal = 1 if p > 0.5 else -1

            # Hour filter: avoid midday solar volatility for negative trades
            is_high_var = h in HIGH_VAR_HOURS_DA_IMB
            high_conf = confidence > 0.2

            # Trade when confident; more selective for negative-signal hours
            trade = 0
            if high_conf:
                if signal > 0:
                    trade = 1  # Structural edge, always trade positive
                elif is_high_var:
                    trade = 1  # Only short during high-variance hours
                elif confidence > 0.4:
                    trade = 1  # Very high confidence override

            # Half-Kelly position, conservative for DA-Imb
            position = kelly_frac * (0.3 + 0.7 * confidence) if trade else 0
            position = min(position, 1.0)  # Cap at 1.0 for this risky spread

            # Stop-loss at -100 EUR/MWh
            raw_pnl = signal * actual * position if trade else 0
            pnl = max(raw_pnl, -100.0 * position) if trade else 0

            if trade:
                recent_pnls.append(pnl)

            results.append({
                'timestamp_hour': row['timestamp_hour'],
                'date': row['date'],
                'hour': h,
                'spread_actual': actual,
                'pred_direction': int(p > 0.5),
                'probability': p,
                'confidence': confidence,
                'position_size': position,
                'signal': signal,
                'trade': trade,
                'pnl_raw': raw_pnl,
                'pnl': pnl,
                'correct': int((signal > 0) == (actual > 0)) if trade else np.nan,
                'strategy': 'B_DA_IMB',
                'month': str(test_month),
            })

    res = pd.DataFrame(results)
    if len(res) == 0:
        print("[!] No results generated")
        return res

    traded = res[res['trade'] == 1]
    print(f"\n[+] Strategy B Results:")
    print(f"    Period: {res['date'].min().date()} to {res['date'].max().date()}")
    print(f"    Total hours: {len(res)}, Traded: {len(traded)} ({len(traded)/len(res)*100:.0f}%)")
    if len(traded) > 0:
        print(f"    Total P&L (with stop-loss): {traded['pnl'].sum():,.0f} EUR")
        print(f"    Total P&L (raw): {traded['pnl_raw'].sum():,.0f} EUR")
        print(f"    Accuracy: {traded['correct'].mean():.1%}")
        print(f"    Avg P&L/trade: {traded['pnl'].mean():.2f} EUR")
        print(f"    Max loss single hour: {traded['pnl'].min():.0f} EUR")
        std_t = traded['pnl'].std()
        print(f"    Sharpe (ann): {traded['pnl'].mean() / std_t * np.sqrt(252*24):.2f}" if std_t > 0 else "    Sharpe: N/A")

        # Structural edge check
        always_long = res.copy()
        always_long['pnl_always'] = always_long['spread_actual']
        print(f"    [*] Structural edge (always DA>IMB): {always_long['pnl_always'].sum():,.0f} EUR total")

        # Feature importance
        feat_imp = pd.Series(clf.feature_importances_, index=avail_feats).sort_values(ascending=False)
        print(f"    Top features: {list(feat_imp.head(7).index)}")

    return res


# ============================================================
# 5. STRATEGY C: IDM-Imbalance with Morning Signal (Enhanced)
# ============================================================

FEAT_C = [
    'hour', 'hour_sin', 'hour_cos', 'dow', 'dow_sin', 'dow_cos',
    'month_sin', 'month_cos', 'is_weekend_int',
    'idm_mean_yest', 'spread_idm_imb_mean_yest',
    'spread_idm_imb_mean_7d', 'spread_idm_imb_mean_3d',
    'spread_idm_imb_mean_momentum',
    'load_fcst_dev', 'load_error_mean_yest',
    'idm_vwap_yest_h', 'spread_idm_imb_yest_h',
    'spread_idm_imb_vol_7d_h',
    'load_fcst_mean',
    # New: production & flow features
    'net_import_yest_h', 'net_import_mean_yest',
    'nuclear_mean_yest', 'solaractual_mean_yest', 'renewable_total_mean_yest',
    'Nuclear_yest_h', 'SolarActual_yest_h', 'renewable_total_yest_h',
]


def run_strategy_c(df):
    """Strategy C: IDM-Imbalance with morning signal + Half-Kelly."""
    print("\n" + "=" * 60)
    print("  STRATEGY C: IDM-Imbalance (Morning Signal + ML, Enhanced)")
    print("=" * 60)

    mask = df['spread_idm_imb'].notna()
    data = df[mask].copy()
    if len(data) == 0:
        print("[!] No IDM-Imbalance spread data available")
        return pd.DataFrame()

    # Build morning signal: mean spread direction for hours 0-5 each day
    morning = data[data['hour'] <= 5].groupby('date').agg(
        morning_spread_mean=('spread_idm_imb', 'mean'),
        morning_spread_pos_pct=('spread_idm_imb', lambda x: (x > 0).mean()),
        morning_n_obs=('spread_idm_imb', 'count'),
    ).reset_index()
    morning['morning_signal'] = (morning['morning_spread_mean'] > 0).astype(int)

    data = data.merge(morning, on='date', how='left')

    data['target_dir'] = (data['spread_idm_imb'] > 0).astype(int)

    morning_feats = ['morning_spread_mean', 'morning_spread_pos_pct', 'morning_signal']
    avail_feats = [f for f in FEAT_C if f in data.columns] + [f for f in morning_feats if f in data.columns]
    print(f"[*] Using {len(avail_feats)} features (incl. morning signal)")

    data['ym'] = data['date'].dt.to_period('M')
    months = sorted(data['ym'].unique())
    min_train_months = 3

    if len(months) <= min_train_months:
        print(f"[!] Only {len(months)} months, need >{min_train_months}")
        return pd.DataFrame()

    results = []
    recent_pnls = []

    for i in range(min_train_months, len(months)):
        test_month = months[i]
        train_months = months[:i]

        train = data[data['ym'].isin(train_months)]
        test = data[data['ym'] == test_month]

        if len(test) == 0:
            continue

        # Only trade hours 6+ (after morning signal available)
        test_tradeable = test[test['hour'] >= 6].copy()
        if len(test_tradeable) == 0:
            continue

        X_train = train[avail_feats].fillna(0)
        y_train_dir = train['target_dir']
        X_test = test_tradeable[avail_feats].fillna(0)

        clf = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.03,
            num_leaves=31, min_child_samples=30,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.2, reg_lambda=0.2,
            verbose=-1, n_jobs=-1
        )
        clf.fit(X_train, y_train_dir)
        proba = clf.predict_proba(X_test)[:, 1]

        kelly_frac = compute_rolling_kelly(np.array(recent_pnls))

        for idx, row in test_tradeable.iterrows():
            j = test_tradeable.index.get_loc(idx)
            p = proba[j]
            actual = row['spread_idm_imb']
            m_sig = row.get('morning_signal', np.nan)

            confidence = abs(p - 0.5) * 2
            signal = 1 if p > 0.5 else -1

            # Morning signal agreement
            morning_agree = False
            if not pd.isna(m_sig):
                morning_agree = ((signal > 0) == (m_sig > 0))

            # Trade when ML confident OR morning agrees
            trade = 1 if (confidence > 0.15 or morning_agree) else 0

            # Half-Kelly with morning agreement bonus
            position = kelly_frac * (0.5 + 0.5 * confidence) if trade else 0
            if morning_agree and trade:
                position = min(position * 1.3, 1.5)
            position = min(position, 1.5)

            pnl = signal * actual * position if trade else 0

            if trade:
                recent_pnls.append(pnl)

            results.append({
                'timestamp_hour': row['timestamp_hour'],
                'date': row['date'],
                'hour': row['hour'],
                'spread_actual': actual,
                'pred_direction': int(p > 0.5),
                'probability': p,
                'confidence': confidence,
                'morning_signal': m_sig,
                'morning_agree': morning_agree,
                'position_size': position,
                'signal': signal,
                'trade': trade,
                'pnl': pnl,
                'correct': int((signal > 0) == (actual > 0)) if trade else np.nan,
                'strategy': 'C_IDM_IMB',
                'month': str(test_month),
            })

    res = pd.DataFrame(results)
    if len(res) == 0:
        print("[!] No results generated")
        return res

    traded = res[res['trade'] == 1]
    print(f"\n[+] Strategy C Results:")
    print(f"    Period: {res['date'].min().date()} to {res['date'].max().date()}")
    print(f"    Total hours: {len(res)}, Traded: {len(traded)} ({len(traded)/len(res)*100:.0f}%)")
    if len(traded) > 0:
        print(f"    Total P&L: {traded['pnl'].sum():,.0f} EUR")
        print(f"    Accuracy: {traded['correct'].mean():.1%}")
        print(f"    Avg P&L/trade: {traded['pnl'].mean():.2f} EUR")
        std_t = traded['pnl'].std()
        print(f"    Sharpe (ann): {traded['pnl'].mean() / std_t * np.sqrt(252*24):.2f}" if std_t > 0 else "    Sharpe: N/A")
        morning_agree_trades = traded[traded['morning_agree'] == True]
        if len(morning_agree_trades) > 0:
            print(f"    Morning-agree trades: {len(morning_agree_trades)}, "
                  f"acc={morning_agree_trades['correct'].mean():.1%}, "
                  f"P&L={morning_agree_trades['pnl'].sum():,.0f} EUR")

        feat_imp = pd.Series(clf.feature_importances_, index=avail_feats).sort_values(ascending=False)
        print(f"    Top features: {list(feat_imp.head(7).index)}")

    return res


# ============================================================
# 6. RULE-BASED BASELINE (Existing, unchanged)
# ============================================================

def run_baseline(df):
    """Replicate the existing rule-based DA-IDM strategy for comparison."""
    print("\n" + "=" * 60)
    print("  BASELINE: Rule-based DA-IDM (from build_da_indicator.py)")
    print("=" * 60)

    mask = df['spread_da_idm'].notna()
    data = df[mask].copy()
    if len(data) == 0:
        print("[!] No data")
        return pd.DataFrame()

    results = []
    for _, row in data.iterrows():
        is_wknd = bool(row.get('is_weekend', False))
        load_dev = row.get('load_fcst_dev', 0) if not pd.isna(row.get('load_fcst_dev', np.nan)) else 0
        spread_yest = row.get('spread_da_idm_yest_h', 0) if not pd.isna(row.get('spread_da_idm_yest_h', np.nan)) else 0
        spread_yest_daily = row.get('spread_da_idm_mean_yest', 0) if not pd.isna(row.get('spread_da_idm_mean_yest', np.nan)) else 0

        actual = row['spread_da_idm']

        if is_wknd:
            signal, conf, trade = 1, 0.89, 1.0
        elif load_dev > 85:
            signal, conf, trade = -1, 0.66, 1.0
        elif abs(spread_yest) >= 20:
            signal = 1 if spread_yest > 0 else -1
            conf, trade = 0.69, 1.0
        elif abs(spread_yest_daily) >= 10:
            signal = 1 if spread_yest_daily > 0 else -1
            conf, trade = 0.58, 0.5
        else:
            signal, conf, trade = 1, 0.54, 0.0

        pnl = signal * actual * trade
        correct = int((signal > 0) == (actual > 0)) if trade > 0 else np.nan

        results.append({
            'timestamp_hour': row['timestamp_hour'],
            'date': row['date'],
            'hour': row['hour'],
            'spread_actual': actual,
            'signal': signal,
            'confidence': conf,
            'trade': trade,
            'position_size': trade,
            'pnl': pnl,
            'correct': correct,
            'strategy': 'Baseline_Rules',
            'month': str(pd.Timestamp(row['date']).to_period('M')),
        })

    res = pd.DataFrame(results)
    traded = res[res['trade'] > 0]
    print(f"\n[+] Baseline Results:")
    print(f"    Period: {res['date'].min().date()} to {res['date'].max().date()}")
    print(f"    Total hours: {len(res)}, Traded: {len(traded)} ({len(traded)/len(res)*100:.0f}%)")
    if len(traded) > 0:
        print(f"    Total P&L: {traded['pnl'].sum():,.0f} EUR")
        print(f"    Accuracy: {traded['correct'].mean():.1%}")
        print(f"    Avg P&L/trade: {traded['pnl'].mean():.2f} EUR")
        sharpe_val = traded['pnl'].mean() / traded['pnl'].std() * np.sqrt(252*24) if traded['pnl'].std() > 0 else 0
        print(f"    Sharpe (ann): {sharpe_val:.2f}")

    return res


# ============================================================
# 7. COMPARISON & REPORTING
# ============================================================

def compute_metrics(res, name):
    """Compute standard metrics for a strategy result DataFrame."""
    if len(res) == 0:
        return {
            'Strategy': name, 'Total P&L': 0, 'Traded Hours': 0,
            'Accuracy': 0, 'Avg P&L/Trade': 0, 'Sharpe': 0,
            'Max Drawdown': 0, 'Win Rate': 0, 'Profitable Months %': 0,
        }

    traded = res[res['trade'] > 0] if 'trade' in res.columns else res
    if len(traded) == 0:
        return {
            'Strategy': name, 'Total P&L': 0, 'Traded Hours': 0,
            'Accuracy': 0, 'Avg P&L/Trade': 0, 'Sharpe': 0,
            'Max Drawdown': 0, 'Win Rate': 0, 'Profitable Months %': 0,
        }

    total_pnl = traded['pnl'].sum()
    cum_pnl = traded['pnl'].cumsum()
    max_dd = (cum_pnl - cum_pnl.cummax()).min()
    std_pnl = traded['pnl'].std()
    sharpe = traded['pnl'].mean() / std_pnl * np.sqrt(252 * 24) if std_pnl > 0 else 0

    correct_col = traded['correct'].dropna()
    accuracy = correct_col.mean() if len(correct_col) > 0 else 0
    win_rate = (traded['pnl'] > 0).mean()

    traded_copy = traded.copy()
    traded_copy['ym'] = pd.to_datetime(traded_copy['date']).dt.to_period('M')
    monthly_pnl = traded_copy.groupby('ym')['pnl'].sum()
    pct_profitable_months = (monthly_pnl > 0).mean() if len(monthly_pnl) > 0 else 0

    # Profit factor
    wins = traded[traded['pnl'] > 0]['pnl'].sum()
    losses = abs(traded[traded['pnl'] <= 0]['pnl'].sum())
    pf = wins / losses if losses > 0 else np.inf

    return {
        'Strategy': name,
        'Total P&L': round(total_pnl, 0),
        'Traded Hours': len(traded),
        'Accuracy': round(accuracy, 3),
        'Avg P&L/Trade': round(traded['pnl'].mean(), 2),
        'Sharpe': round(sharpe, 2),
        'Max Drawdown': round(max_dd, 0),
        'Win Rate': round(win_rate, 3),
        'Profit Factor': round(pf, 2),
        'Profitable Months %': round(pct_profitable_months, 2),
    }


def print_comparison(metrics_list):
    """Print formatted comparison table."""
    print("\n" + "=" * 90)
    print("  STRATEGY COMPARISON TABLE")
    print("=" * 90)

    comp = pd.DataFrame(metrics_list)
    fmt_map = {
        'Total P&L': lambda x: f"{x:>10,.0f} EUR",
        'Traded Hours': lambda x: f"{x:>8,.0f}",
        'Accuracy': lambda x: f"{x:>8.1%}",
        'Avg P&L/Trade': lambda x: f"{x:>8.2f} EUR",
        'Sharpe': lambda x: f"{x:>8.2f}",
        'Max Drawdown': lambda x: f"{x:>10,.0f} EUR",
        'Win Rate': lambda x: f"{x:>8.1%}",
        'Profit Factor': lambda x: f"{x:>8.2f}",
        'Profitable Months %': lambda x: f"{x:>8.0%}",
    }

    for _, row in comp.iterrows():
        print(f"\n--- {row['Strategy']} ---")
        for col, fmt in fmt_map.items():
            if col in row:
                print(f"  {col:25s}: {fmt(row[col])}")

    print("\n" + "-" * 100)
    print(f"\n{'Strategy':<30s} {'P&L':>12s} {'Acc':>8s} {'Sharpe':>8s} {'MaxDD':>12s} {'WinRate':>8s} {'PF':>6s}")
    print("-" * 84)
    for _, row in comp.iterrows():
        print(f"{row['Strategy']:<30s} {row['Total P&L']:>10,.0f}E {row['Accuracy']:>7.1%} {row['Sharpe']:>8.2f} {row['Max Drawdown']:>10,.0f}E {row['Win Rate']:>7.1%} {row['Profit Factor']:>6.2f}")

    return comp


# ============================================================
# 8. CHARTS
# ============================================================

def plot_results(all_results, metrics_df):
    """Generate comparison charts."""
    print("[*] Generating charts...")

    colors = {'Baseline_Rules': '#9e9e9e', 'A_DA_IDM': '#1976d2',
              'B_DA_IMB': '#d32f2f', 'C_IDM_IMB': '#4caf50',
              'Always_DA>IMB': '#ef9a9a', 'Always_IDM>IMB': '#a5d6a7',
              'Portfolio': '#ff9800'}
    labels = {'Baseline_Rules': 'Baseline (Rules)',
              'A_DA_IDM': 'A: DA-IDM (ML)',
              'B_DA_IMB': 'B: DA-Imbalance (ML)',
              'C_IDM_IMB': 'C: IDM-Imbalance (ML)',
              'Always_DA>IMB': 'B0: Always DA>IMB',
              'Always_IDM>IMB': 'C0: Always IDM>IMB',
              'Portfolio': 'Portfolio (A+B+C)'}

    jan26 = pd.Timestamp('2026-01-01')

    # Chart 1: Cumulative P&L - main strategies only
    fig, ax = plt.subplots(figsize=(14, 7))
    main_strats = ['Baseline_Rules', 'A_DA_IDM', 'B_DA_IMB', 'C_IDM_IMB', 'Portfolio']
    for strat in all_results['strategy'].unique():
        if strat not in main_strats:
            continue
        sub = all_results[(all_results['strategy'] == strat) & (all_results['trade'] > 0)].copy()
        if len(sub) == 0:
            continue
        sub = sub.sort_values('timestamp_hour')
        sub['cum_pnl'] = sub['pnl'].cumsum()
        ax.plot(sub['timestamp_hour'], sub['cum_pnl'],
                color=colors.get(strat, 'gray'), lw=1.5,
                label=f"{labels.get(strat, strat)} ({sub['pnl'].sum():,.0f} EUR)")

    ax.axhline(0, color='black', lw=0.5)
    if all_results['timestamp_hour'].max() > jan26:
        ax.axvline(jan26, color='red', ls='--', alpha=0.3, label='2026')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative P&L (EUR)')
    ax.set_title('Spread Strategies - Cumulative P&L (Enhanced with Production Features + Half-Kelly)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / '01_cumulative_pnl_comparison.png', dpi=150)
    plt.close(fig)

    # Chart 2: Monthly P&L by strategy
    plot_strats = [s for s in main_strats if s in all_results['strategy'].unique()]
    n_strats = len(plot_strats)
    if n_strats > 0:
        n_rows = (n_strats + 1) // 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(16, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        for i, strat in enumerate(plot_strats):
            ax = axes[i // 2][i % 2]
            sub = all_results[(all_results['strategy'] == strat) & (all_results['trade'] > 0)].copy()
            if len(sub) == 0:
                ax.set_title(f"{labels.get(strat, strat)} - No Data")
                continue
            sub['ym'] = pd.to_datetime(sub['date']).dt.to_period('M')
            monthly = sub.groupby('ym')['pnl'].sum()
            x = range(len(monthly))
            bar_colors = ['#d32f2f' if str(m).startswith('2026') else '#1976d2' for m in monthly.index]
            ax.bar(x, monthly.values, color=bar_colors, edgecolor='white')
            ax.set_xticks(list(x))
            ax.set_xticklabels([str(m) for m in monthly.index], rotation=45, ha='right', fontsize=7)
            ax.set_title(labels.get(strat, strat))
            ax.set_ylabel('Monthly P&L (EUR)')
            ax.axhline(0, color='black', lw=0.5)
            ax.grid(True, alpha=0.3)
        # Hide unused subplots
        for j in range(len(plot_strats), n_rows * 2):
            axes[j // 2][j % 2].set_visible(False)
        fig.suptitle('Monthly P&L (red = 2026, blue = 2025)', fontweight='bold')
        fig.tight_layout()
        fig.savefig(OUT / '02_monthly_pnl_by_strategy.png', dpi=150)
        plt.close(fig)

    # Chart 3: ML vs Naive benchmarks
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # B vs B0
    ax = axes[0]
    for strat, color in [('B_DA_IMB', '#d32f2f'), ('Always_DA>IMB', '#ef9a9a')]:
        sub = all_results[(all_results['strategy'] == strat) & (all_results['trade'] > 0)].copy()
        if len(sub) == 0:
            continue
        sub = sub.sort_values('timestamp_hour')
        sub['cum_pnl'] = sub['pnl'].cumsum()
        ax.plot(sub['timestamp_hour'], sub['cum_pnl'], color=color, lw=1.5,
                label=f"{labels.get(strat, strat)} ({sub['pnl'].sum():,.0f} EUR)")
    ax.set_title('DA-Imbalance: ML vs Always-Positive')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # C vs C0
    ax = axes[1]
    for strat, color in [('C_IDM_IMB', '#4caf50'), ('Always_IDM>IMB', '#a5d6a7')]:
        sub = all_results[(all_results['strategy'] == strat) & (all_results['trade'] > 0)].copy()
        if len(sub) == 0:
            continue
        sub = sub.sort_values('timestamp_hour')
        sub['cum_pnl'] = sub['pnl'].cumsum()
        ax.plot(sub['timestamp_hour'], sub['cum_pnl'], color=color, lw=1.5,
                label=f"{labels.get(strat, strat)} ({sub['pnl'].sum():,.0f} EUR)")
    ax.set_title('IDM-Imbalance: ML vs Always-Positive')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle('ML Alpha vs Structural Edge', fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / '03_ml_vs_naive_benchmarks.png', dpi=150)
    plt.close(fig)

    # Chart 4: Strategy comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    n = len(metrics_df)
    x = range(n)
    bar_colors = [colors.get(s.split(':')[0].split('(')[0].strip().replace(' ', '_'), '#9e9e9e')
                  for s in metrics_df['Strategy']]
    # Use simple cycling colors instead
    bar_colors = ['#9e9e9e', '#1976d2', '#d32f2f', '#ef9a9a', '#4caf50', '#a5d6a7', '#ff9800'][:n]

    ax = axes[0]
    ax.barh(x, metrics_df['Total P&L'], color=bar_colors, edgecolor='white')
    ax.set_yticks(list(x))
    ax.set_yticklabels(metrics_df['Strategy'], fontsize=7)
    ax.set_title('Total P&L (EUR)')
    ax.axvline(0, color='black', lw=0.5)

    ax = axes[1]
    ax.barh(x, metrics_df['Accuracy'], color=bar_colors, edgecolor='white')
    ax.set_yticks(list(x))
    ax.set_yticklabels(metrics_df['Strategy'], fontsize=7)
    ax.set_title('Direction Accuracy')
    ax.axvline(0.5, color='red', ls='--', alpha=0.5)

    ax = axes[2]
    ax.barh(x, metrics_df['Sharpe'], color=bar_colors, edgecolor='white')
    ax.set_yticks(list(x))
    ax.set_yticklabels(metrics_df['Strategy'], fontsize=7)
    ax.set_title('Annualized Sharpe')
    ax.axvline(0, color='black', lw=0.5)

    fig.suptitle('Strategy Performance Comparison', fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / '04_strategy_comparison_bars.png', dpi=150)
    plt.close(fig)

    # Chart 5: Rolling accuracy
    fig, ax = plt.subplots(figsize=(14, 6))
    for strat in main_strats:
        sub = all_results[(all_results['strategy'] == strat) & (all_results['trade'] > 0)].copy()
        if len(sub) == 0:
            continue
        sub = sub.sort_values('timestamp_hour')
        sub['roll_acc'] = sub['correct'].rolling(500, min_periods=100).mean()
        ax.plot(sub['timestamp_hour'], sub['roll_acc'],
                color=colors.get(strat, 'gray'), lw=1.2,
                label=labels.get(strat, strat))
    ax.axhline(0.5, color='red', ls='--', alpha=0.5, label='Random')
    if all_results['timestamp_hour'].max() > jan26:
        ax.axvline(jan26, color='red', ls='--', alpha=0.3)
    ax.set_xlabel('Date')
    ax.set_ylabel('Rolling 500-trade Accuracy')
    ax.set_title('Strategy Accuracy Over Time')
    ax.legend(fontsize=8)
    ax.set_ylim(0.35, 0.80)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / '05_rolling_accuracy.png', dpi=150)
    plt.close(fig)

    print(f"[+] Charts saved to {OUT}")


# ============================================================
# 9. YEAR-OVER-YEAR + STRUCTURAL BENCHMARKS
# ============================================================

def year_analysis(all_results):
    """Print year-over-year performance for each strategy."""
    print("\n" + "=" * 70)
    print("  YEAR-OVER-YEAR ANALYSIS")
    print("=" * 70)

    all_results_copy = all_results.copy()
    all_results_copy['year'] = pd.to_datetime(all_results_copy['date']).dt.year

    for strat in all_results_copy['strategy'].unique():
        sub = all_results_copy[(all_results_copy['strategy'] == strat) & (all_results_copy['trade'] > 0)]
        if len(sub) == 0:
            continue
        print(f"\n--- {strat} ---")
        for yr in sorted(sub['year'].unique()):
            yr_data = sub[sub['year'] == yr]
            correct_vals = yr_data['correct'].dropna()
            acc = correct_vals.mean() if len(correct_vals) > 0 else 0
            pnl = yr_data['pnl'].sum()
            n = len(yr_data)
            avg = yr_data['pnl'].mean()
            print(f"  {yr}: P&L={pnl:>10,.0f} EUR, Acc={acc:.1%}, n={n:>5}, Avg={avg:>6.1f} EUR/trade")


def build_always_positive_benchmark(df, spread_col, name, ml_res):
    """Build 'always positive' benchmark for same period as ML strategy."""
    if len(ml_res) == 0:
        return pd.DataFrame()

    ml_dates = pd.to_datetime(ml_res['date']).unique()
    min_date = ml_dates.min()
    max_date = ml_dates.max()

    mask = (df[spread_col].notna() &
            (df['date'] >= min_date) &
            (df['date'] <= max_date))
    data = df[mask].copy()

    if len(data) == 0:
        return pd.DataFrame()

    results = []
    for _, row in data.iterrows():
        actual = row[spread_col]
        results.append({
            'timestamp_hour': row['timestamp_hour'],
            'date': row['date'],
            'hour': row['hour'],
            'spread_actual': actual,
            'signal': 1,
            'trade': 1,
            'position_size': 1.0,
            'pnl': actual,
            'correct': int(actual > 0),
            'strategy': name,
            'month': str(pd.Timestamp(row['date']).to_period('M')),
            'confidence': 1.0,
        })

    res = pd.DataFrame(results)
    traded = res[res['trade'] > 0]
    print(f"\n[+] {name} Benchmark:")
    print(f"    Period: {res['date'].min().date()} to {res['date'].max().date()}")
    print(f"    Total hours: {len(traded)}")
    print(f"    Total P&L: {traded['pnl'].sum():,.0f} EUR")
    print(f"    Accuracy (pos rate): {traded['correct'].mean():.1%}")
    print(f"    Avg P&L/trade: {traded['pnl'].mean():.2f} EUR")
    return res


# ============================================================
# 10. MULTI-SPREAD PORTFOLIO
# ============================================================

def build_portfolio(res_a, res_b, res_c):
    """Combine all strategies into equal-weight portfolio with diversification."""
    print("\n" + "=" * 60)
    print("  PORTFOLIO: Combined Multi-Spread (Equal Weight)")
    print("=" * 60)

    all_traded = []
    for res, weight in [(res_a, 1/3), (res_b, 1/3), (res_c, 1/3)]:
        if len(res) == 0:
            continue
        traded = res[res['trade'] > 0].copy()
        if len(traded) == 0:
            continue
        traded['pnl_weighted'] = traded['pnl'] * weight
        traded['strategy_orig'] = traded['strategy']
        all_traded.append(traded[['timestamp_hour', 'date', 'hour', 'pnl_weighted',
                                  'correct', 'strategy_orig']])

    if not all_traded:
        print("[!] No trades to combine")
        return pd.DataFrame()

    combined = pd.concat(all_traded, ignore_index=True)

    # Aggregate by timestamp_hour (sum P&L from concurrent strategies)
    portfolio = combined.groupby(['timestamp_hour', 'date', 'hour']).agg(
        pnl=('pnl_weighted', 'sum'),
        correct=('correct', 'mean'),  # Average accuracy across strategies
        n_strats=('strategy_orig', 'count'),
    ).reset_index()
    portfolio['trade'] = 1
    portfolio['strategy'] = 'Portfolio'
    portfolio['signal'] = 1
    portfolio['position_size'] = 1.0
    portfolio['confidence'] = 1.0
    portfolio['month'] = pd.to_datetime(portfolio['date']).dt.to_period('M').astype(str)

    # Correct direction: if portfolio pnl > 0, mark as correct
    portfolio['correct'] = (portfolio['pnl'] > 0).astype(float)

    print(f"[+] Portfolio Results:")
    print(f"    Period: {portfolio['date'].min()} to {portfolio['date'].max()}")
    print(f"    Total P&L: {portfolio['pnl'].sum():,.0f} EUR")
    print(f"    Avg trades/hour: {portfolio['n_strats'].mean():.1f}")
    print(f"    Win rate: {(portfolio['pnl'] > 0).mean():.1%}")
    std_t = portfolio['pnl'].std()
    if std_t > 0:
        print(f"    Sharpe (ann): {portfolio['pnl'].mean() / std_t * np.sqrt(252*24):.2f}")

    # Diversification benefit
    daily = portfolio.groupby('date')['pnl'].sum()
    dd = (daily.cumsum() - daily.cumsum().cummax()).min()
    print(f"    Max drawdown (daily): {dd:,.0f} EUR")

    return portfolio


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("  SPECULATIVE SPREAD TRADING STRATEGIES (ENHANCED)")
    print("  Slovak Electricity Market")
    print("  Features: Production/Renewable, Half-Kelly, Hour-Filter, Portfolio")
    print("=" * 70)

    # 1. Load data (now includes production/renewable)
    df = load_data()

    # 2. Build features (now includes nuclear, solar, gas, flows)
    df = build_features(df)

    # 3. Run all strategies (enhanced)
    res_baseline = run_baseline(df)
    res_a = run_strategy_a(df)
    res_b = run_strategy_b(df)
    res_c = run_strategy_c(df)

    # 4. Structural benchmarks
    res_b_always = build_always_positive_benchmark(df, 'spread_da_imb', 'Always_DA>IMB', res_b)
    res_c_always = build_always_positive_benchmark(df, 'spread_idm_imb', 'Always_IDM>IMB', res_c)

    # 5. Multi-spread portfolio
    res_portfolio = build_portfolio(res_a, res_b, res_c)

    # 6. Compute metrics
    metrics = []
    for res, name in [
        (res_baseline, 'Baseline (Rules)'),
        (res_a, 'A: DA-IDM (ML)'),
        (res_b, 'B: DA-Imb (ML+Risk)'),
        (res_b_always, 'B0: Always DA>IMB'),
        (res_c, 'C: IDM-Imb (Morning+ML)'),
        (res_c_always, 'C0: Always IDM>IMB'),
        (res_portfolio, 'Portfolio (A+B+C)'),
    ]:
        if len(res) > 0:
            metrics.append(compute_metrics(res, name))

    # 7. Comparison table
    comp = print_comparison(metrics)

    # 8. Combine all results
    all_dfs = [r for r in [res_baseline, res_a, res_b, res_b_always, res_c, res_c_always, res_portfolio]
               if len(r) > 0]
    all_results = pd.concat(all_dfs, ignore_index=True)
    all_results['timestamp_hour'] = pd.to_datetime(all_results['timestamp_hour'])
    all_results['date'] = pd.to_datetime(all_results['date'])

    # 9. Year-over-year
    year_analysis(all_results)

    # 10. Charts
    plot_results(all_results, comp)

    # 11. Save results
    all_results.to_csv(OUT / 'spread_results.csv', index=False)
    comp.to_csv(OUT / 'strategy_comparison.csv', index=False)
    print(f"\n[+] Results saved to {OUT / 'spread_results.csv'}")
    print(f"[+] Comparison saved to {OUT / 'strategy_comparison.csv'}")

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
