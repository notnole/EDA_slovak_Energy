"""
Enhanced BESS Scheduling Strategy
==================================
1 MW / 2 MWh battery, 2 cycles/day, persistent SoC across days.

Improvements over baseline (192,322 EUR, 78% capture):
  1. ML-predicted price ranks using LightGBM walk-forward
  2. DA price forecasts integrated (Sep 2025+, r=0.795 with actual)
  3. Richer feature set: load shape, yesterday prices, weekly patterns,
     cross-border flows, calendar effects
  4. Degradation-aware DP optimizer: per-cycle cost subtracted from spread,
     so low-spread days naturally get fewer or no cycles
  5. Adaptive mode selection: compares 1-cycle vs 2-cycle after degradation

Walk-forward validation: expanding window, predict next month.

Usage:
    python enhanced_bess_strategy.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("[!] LightGBM not available, using fallback rank ensemble")

BASE = Path(r'C:\Users\20254757\pycharmprojects\ipesoft_eda_data')
OUT = BASE / 'MarketPriceGap' / 'strategies' / 'bess'


# ============================================================
# DP OPTIMIZER (degradation-aware)
# ============================================================

# Battery degradation cost: ~200 EUR/kWh replacement, 5000 cycle lifetime
# For 2 MWh battery = 400k EUR replacement, over 5000 cycles = 80 EUR/cycle
# But actual degradation depends on DoD, temperature, etc.
# Conservative estimate: 50 EUR per full cycle (charge 2 MWh + discharge 2 MWh)
# This translates to 25 EUR per charge action (1 MWh) in the DP
DEGRADATION_COST_PER_MWH = 25.0  # EUR per MWh charged or discharged


def optimize_bess_dp(prices, predicted_ranks, n_charge=4, n_discharge=4,
                     start_soc=1, terminal_penalty=20.0, efficiency=1.0,
                     degradation_cost=0.0):
    """
    Optimal BESS scheduling via dynamic programming.

    Parameters
    ----------
    degradation_cost : float
        Cost per MWh of charging/discharging (accounts for battery wear).
        Applied to predicted_ranks during optimization so the DP naturally
        avoids low-spread trades.
    """
    n_hours = len(prices)
    if n_hours == 0 or np.any(np.isnan(prices)):
        return np.zeros(n_hours), 0.0, np.ones(n_hours + 1) * start_soc

    p = np.array(predicted_ranks[:n_hours], dtype=float)
    actual = np.array(prices[:n_hours], dtype=float)

    NC = n_charge + 1
    ND = n_discharge + 1
    INF = -1e18

    dp = np.full((n_hours + 1, 3, NC, ND), INF)
    choice = np.zeros((n_hours, 3, NC, ND), dtype=int)

    for s in range(3):
        penalty = 0.0
        if s == 0 or s == 2:
            penalty = -terminal_penalty
        for cr in range(NC):
            for dr in range(ND):
                dp[n_hours][s][cr][dr] = penalty

    for h in range(n_hours - 1, -1, -1):
        for s in range(3):
            for cr in range(NC):
                for dr in range(ND):
                    best = INF
                    best_a = 0

                    val = dp[h + 1][s][cr][dr]
                    if val > best:
                        best = val
                        best_a = 0

                    # Charge: pay price + degradation cost
                    if cr > 0 and s + 1 <= 2:
                        val = -(p[h] + degradation_cost) + dp[h + 1][s + 1][cr - 1][dr]
                        if val > best:
                            best = val
                            best_a = 1

                    # Discharge: earn price - degradation cost
                    if dr > 0 and s - 1 >= 0:
                        val = (p[h] * efficiency - degradation_cost) + dp[h + 1][s - 1][cr][dr - 1]
                        if val > best:
                            best = val
                            best_a = -1

                    dp[h][s][cr][dr] = best
                    choice[h][s][cr][dr] = best_a

    actions = np.zeros(n_hours)
    soc_trace = np.zeros(n_hours + 1)
    soc_trace[0] = start_soc
    soc = start_soc
    cr = n_charge
    dr = n_discharge

    for h in range(n_hours):
        a = choice[h][soc][cr][dr]
        actions[h] = a
        if a == 1:
            soc += 1
            cr -= 1
        elif a == -1:
            soc -= 1
            dr -= 1
        soc_trace[h + 1] = soc

    # Revenue always based on ACTUAL prices (no degradation in revenue calc)
    revenue = 0.0
    for h in range(n_hours):
        if actions[h] == 1:
            revenue -= actual[h]
        elif actions[h] == -1:
            revenue += actual[h] * efficiency

    return actions, revenue, soc_trace


def optimize_bess_adaptive(prices, predicted_ranks, start_soc=1,
                           terminal_penalty=20.0, degradation_cost=0.0):
    """
    Adaptive cycle selection: run DP with both 1-cycle and 2-cycle limits
    using ranks (no degradation in DP since ranks are not EUR).
    Then compare net revenue (actual EUR - degradation) to pick best mode.

    The DP always optimizes using predicted_ranks with NO degradation cost
    (since ranks are ordinal, not EUR). Degradation is applied post-hoc
    to the actual EUR revenue to decide which cycle mode is best.

    Uses a higher terminal penalty (40 EUR) to strongly discourage SoC drift,
    since ending at SoC=0 or SoC=2 reduces optionality for the next day.

    Returns the best (actions, revenue, soc_trace, n_cycles_used).
    """
    # Use a stronger terminal penalty for the adaptive DP to prevent SoC drift
    strong_penalty = max(terminal_penalty, 40.0)

    # 2-cycle mode: 4 charge + 4 discharge (DP uses ranks, no deg cost)
    act2, rev2, soc2 = optimize_bess_dp(
        prices, predicted_ranks, n_charge=4, n_discharge=4,
        start_soc=start_soc, terminal_penalty=strong_penalty,
        degradation_cost=0.0
    )
    n_charge2 = int(np.sum(act2 > 0.5))
    n_discharge2 = int(np.sum(act2 < -0.5))
    end_soc2 = int(soc2[-1])

    # 1-cycle mode: 2 charge + 2 discharge
    act1, rev1, soc1 = optimize_bess_dp(
        prices, predicted_ranks, n_charge=2, n_discharge=2,
        start_soc=start_soc, terminal_penalty=strong_penalty,
        degradation_cost=0.0
    )
    n_charge1 = int(np.sum(act1 > 0.5))
    n_discharge1 = int(np.sum(act1 < -0.5))
    end_soc1 = int(soc1[-1])

    # Compute net revenue after degradation for comparison
    deg2 = degradation_cost * (n_charge2 + n_discharge2)
    deg1 = degradation_cost * (n_charge1 + n_discharge1)
    net2 = rev2 - deg2
    net1 = rev1 - deg1

    # Also consider idle (0 cycles) - just carry SoC forward
    n_hours = len(prices)
    net0 = 0.0

    # Pick best mode by net revenue after degradation
    if net2 >= net1 and net2 >= net0:
        return act2, rev2, soc2, 2
    elif net1 >= net0:
        return act1, rev1, soc1, 1
    else:
        # Even 1 cycle is unprofitable after degradation -> idle
        return np.zeros(n_hours), 0.0, np.ones(n_hours + 1) * start_soc, 0


def naive_schedule(n_hours, start_soc):
    """Naive 2-cycle fixed schedule: charge h2-3, discharge h10-11, charge h14-15, discharge h18-19."""
    actions = np.zeros(n_hours)
    if n_hours < 20:
        return actions, start_soc

    schedule = {2: 1, 3: 1, 10: -1, 11: -1, 14: 1, 15: 1, 18: -1, 19: -1}
    soc = start_soc
    for h in range(n_hours):
        a = schedule.get(h, 0)
        new_soc = soc + a
        if new_soc < 0 or new_soc > 2:
            a = 0
        actions[h] = a
        soc += a
    return actions, soc


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """Load all data sources and merge into single hourly DataFrame."""
    print("[*] Loading market prices...")
    mkt = pd.read_csv(BASE / 'MarketPriceGap' / 'data' / 'processed' / 'hourly_market_prices.csv')
    mkt['timestamp_hour'] = pd.to_datetime(mkt['timestamp_hour'])
    mkt['date'] = pd.to_datetime(mkt['date'])
    if mkt['is_weekend'].dtype == object:
        mkt['is_weekend'] = mkt['is_weekend'].str.lower().isin(['true', '1'])
    else:
        mkt['is_weekend'] = mkt['is_weekend'].astype(bool)
    print(f"[+] Market: {len(mkt)} rows, {mkt['date'].min().date()} to {mkt['date'].max().date()}")

    # Load forecasts
    print("[*] Loading load forecasts...")
    load = pd.read_csv(BASE / 'features' / 'DamasLoad' / 'load_data.csv')
    load['timestamp_hour'] = pd.to_datetime(load['datetime'])
    df = mkt.merge(load[['timestamp_hour', 'actual_load_mw', 'forecast_load_mw', 'forecast_error_mw']],
                   on='timestamp_hour', how='left')
    print(f"[+] Load data merged: {df['forecast_load_mw'].notna().sum()} non-null load forecasts")

    # Load DA price forecasts
    print("[*] Loading DA price forecasts...")
    fcst_path = BASE / 'data' / 'clean' / 'market' / 'da_forecasts' / 'da_price_forecasts_15min.csv'
    if fcst_path.exists():
        fcst = pd.read_csv(fcst_path)
        fcst['datetime'] = pd.to_datetime(fcst['datetime'])
        fcst['date'] = fcst['datetime'].dt.normalize()
        fcst['hour'] = fcst['datetime'].dt.hour

        # Aggregate 15-min forecasts to hourly
        fcst_h = fcst.groupby(['date', 'hour']).agg(
            da_fcst1=('sk_forecast1_spot', 'mean'),
            da_fcst2=('sk_forecast2_spot', 'mean'),
        ).reset_index()
        fcst_h['date'] = pd.to_datetime(fcst_h['date'])

        df = df.merge(fcst_h, on=['date', 'hour'], how='left')
        n_fcst = df['da_fcst1'].notna().sum()
        print(f"[+] DA forecasts merged: {n_fcst} non-null hourly rows")
    else:
        print("[-] DA price forecasts not found")
        df['da_fcst1'] = np.nan
        df['da_fcst2'] = np.nan

    # Load production data (nuclear, solar)
    print("[*] Loading production data...")
    try:
        import re
        prod_path = BASE / 'RawData' / 'ProductionPerType.csv'
        if prod_path.exists():
            col_names = ['Biomass', 'FosilOil', 'HardCoal', 'HydroPump',
                         'HydroPumpConsumption', 'HydroReservoir', 'HydroRunRiver',
                         'Lignite', 'NatGas', 'Nuclear', 'Other', 'OtherRenewable',
                         'SolarActual', 'SolarForecast']
            dt_pat = re.compile(r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}')
            records = []
            with open(prod_path, 'r', encoding='utf-8-sig') as f:
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
            if 'Nuclear' in prod.columns:
                df = df.merge(
                    prod[['timestamp_hour', 'Nuclear', 'NatGas', 'SolarActual']].drop_duplicates('timestamp_hour'),
                    on='timestamp_hour', how='left'
                )
                print(f"[+] Production merged: {df['Nuclear'].notna().sum()} nuclear observations")
            else:
                df['Nuclear'] = np.nan
                df['NatGas'] = np.nan
                df['SolarActual'] = np.nan
        else:
            df['Nuclear'] = np.nan
            df['NatGas'] = np.nan
            df['SolarActual'] = np.nan
    except Exception as e:
        print(f"[-] Production parse failed: {e}")
        df['Nuclear'] = np.nan
        df['NatGas'] = np.nan
        df['SolarActual'] = np.nan

    df = df.sort_values(['date', 'hour']).reset_index(drop=True)
    print(f"[+] Final dataset: {len(df)} rows, {df.shape[1]} columns")
    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(df):
    """
    Build features available D-1 for price rank prediction.
    All features respect information timing.
    """
    print("[*] Building features for rank prediction...")

    # --- Daily aggregates (from previous days) ---
    daily = df.groupby('date').agg(
        da_mean=('da_price', 'mean'),
        da_max=('da_price', 'max'),
        da_min=('da_price', 'min'),
        da_std=('da_price', 'std'),
        da_range=('da_price', lambda x: x.max() - x.min()),
        load_fcst_mean=('forecast_load_mw', 'mean'),
        load_actual_mean=('actual_load_mw', 'mean'),
        nuclear_mean=('Nuclear', 'mean'),
    ).reset_index()
    daily = daily.sort_values('date').reset_index(drop=True)

    # Yesterday's values
    for col in ['da_mean', 'da_max', 'da_min', 'da_std', 'da_range',
                'load_actual_mean', 'nuclear_mean']:
        daily[f'{col}_yest'] = daily[col].shift(1)

    # 7-day rolling averages
    for col in ['da_mean', 'da_range', 'load_actual_mean']:
        daily[f'{col}_7d'] = daily[col].shift(1).rolling(7, min_periods=3).mean()

    # Load forecast deviation from recent actual
    daily['load_fcst_dev'] = daily['load_fcst_mean'] - daily['load_actual_mean_7d']

    # Merge back
    daily_feats = daily.drop(columns=[], errors='ignore')
    df = df.merge(daily_feats, on='date', how='left', suffixes=('', '_daily'))

    # --- Hourly lagged features (same hour, previous days) ---
    df = df.sort_values(['hour', 'date']).reset_index(drop=True)
    df['da_price_yest_h'] = df.groupby('hour')['da_price'].shift(1)
    df['da_price_2d_h'] = df.groupby('hour')['da_price'].shift(2)
    df['da_price_7d_h'] = df.groupby('hour')['da_price'].shift(7)
    df['da_price_14d_h'] = df.groupby('hour')['da_price'].shift(14)

    # IDM and imbalance lagged (same hour yesterday)
    df['idm_vwap_yest_h'] = df.groupby('hour')['idm_vwap'].shift(1)
    df['imb_price_yest_h'] = df.groupby('hour')['imb_price'].shift(1)

    df = df.sort_values(['date', 'hour']).reset_index(drop=True)

    # --- Within-day rank features (available D-1) ---
    # Load forecast rank within day
    df['rank_load_fcst'] = df.groupby('date')['forecast_load_mw'].rank(method='first', na_option='bottom')

    # Yesterday price rank
    df['rank_yest_price'] = df.groupby('date')['da_price_yest_h'].rank(method='first', na_option='bottom')

    # 7-day price rank
    df['rank_7d_price'] = df.groupby('date')['da_price_7d_h'].rank(method='first', na_option='bottom')

    # DA forecast rank (when available)
    df['rank_da_fcst1'] = df.groupby('date')['da_fcst1'].rank(method='first', na_option='bottom')
    df['rank_da_fcst2'] = df.groupby('date')['da_fcst2'].rank(method='first', na_option='bottom')

    # Net import rank
    # We can't use today's actual net_import, but yesterday's same-hour is available
    df = df.sort_values(['hour', 'date']).reset_index(drop=True)
    df['net_import_yest_h'] = df.groupby('hour')['net_import'].shift(1)
    df = df.sort_values(['date', 'hour']).reset_index(drop=True)
    df['rank_net_import_yest'] = df.groupby('date')['net_import_yest_h'].rank(method='first', na_option='bottom')

    # Calendar features
    df['dow'] = df['date'].dt.dayofweek
    df['month_feat'] = df['date'].dt.month
    df['is_weekend_int'] = df['is_weekend'].astype(int)

    # Actual rank (target for training)
    df['rank_actual'] = df.groupby('date')['da_price'].rank(method='first', na_option='bottom')

    # Baseline predicted rank (for comparison)
    df['rank_baseline'] = (
        0.40 * df['rank_load_fcst'].fillna(12) +
        0.35 * df['rank_yest_price'].fillna(12) +
        0.25 * df['rank_7d_price'].fillna(12)
    )

    print(f"[+] Features built: {df.shape[1]} columns")
    return df


# ============================================================
# ML RANK PREDICTOR
# ============================================================

FEATURE_COLS = [
    'hour',
    'dow',
    'month_feat',
    'is_weekend_int',
    # Load features
    'forecast_load_mw',
    'rank_load_fcst',
    'load_fcst_dev',
    'load_fcst_mean',
    'load_actual_mean_yest',
    # Yesterday price features
    'da_price_yest_h',
    'da_price_2d_h',
    'da_price_7d_h',
    'rank_yest_price',
    'rank_7d_price',
    # Daily aggregates from yesterday
    'da_mean_yest',
    'da_max_yest',
    'da_min_yest',
    'da_std_yest',
    'da_range_yest',
    'da_mean_7d',
    'da_range_7d',
    # Cross-border
    'net_import_yest_h',
    'rank_net_import_yest',
    # Production
    'nuclear_mean_yest',
    # IDM/imbalance (yesterday same hour)
    'idm_vwap_yest_h',
    'imb_price_yest_h',
    # DA forecasts (when available)
    'da_fcst1',
    'da_fcst2',
    'rank_da_fcst1',
    'rank_da_fcst2',
]


def train_lightgbm_walk_forward(df):
    """
    Walk-forward LightGBM training for price rank prediction.
    Expanding window: train on all data up to month M, predict month M+1.
    Falls back to baseline rank where training data is insufficient.
    """
    if not HAS_LIGHTGBM:
        print("[-] LightGBM not available, skipping ML training")
        return df

    print("[*] Training LightGBM walk-forward rank predictor...")

    df['year_month'] = df['date'].dt.to_period('M')
    all_months = sorted(df['year_month'].unique())

    # Need at least 3 months for training
    min_train_months = 3

    # LightGBM params for rank prediction
    lgb_params = {
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
    }

    df['rank_ml'] = np.nan
    total_predicted = 0

    for i in range(min_train_months, len(all_months)):
        pred_month = all_months[i]
        train_months = all_months[:i]

        train_mask = df['year_month'].isin(train_months)
        pred_mask = df['year_month'] == pred_month

        train_df = df[train_mask].copy()
        pred_df = df[pred_mask].copy()

        if len(pred_df) == 0:
            continue

        # Determine which features are available
        # DA forecasts may not be available for early months
        available_features = []
        for col in FEATURE_COLS:
            if col in train_df.columns:
                # Check if feature has enough non-null values in training
                non_null_frac = train_df[col].notna().mean()
                if non_null_frac > 0.1:
                    available_features.append(col)

        # For prediction, check which features are available
        pred_available = []
        for col in available_features:
            pred_non_null = pred_df[col].notna().mean()
            if pred_non_null > 0.3:
                pred_available.append(col)

        use_features = pred_available

        if len(use_features) < 5:
            continue

        # Prepare training data
        target = 'rank_actual'
        X_train = train_df[use_features].copy()
        y_train = train_df[target].copy()

        # Drop rows with missing target
        valid = y_train.notna()
        X_train = X_train[valid]
        y_train = y_train[valid]

        if len(X_train) < 500:
            continue

        X_pred = pred_df[use_features].copy()

        # Fill NaN with -999 for LightGBM (it handles NaN natively)
        X_train = X_train.fillna(-999)
        X_pred = X_pred.fillna(-999)

        # Train
        train_data = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=300,
            valid_sets=[train_data],
            callbacks=[lgb.log_evaluation(0)],
        )

        # Predict
        preds = model.predict(X_pred)
        df.loc[pred_mask, 'rank_ml'] = preds
        total_predicted += len(pred_df)

        # Quick evaluation
        pred_with_actual = pred_df[['date', 'hour']].copy()
        pred_with_actual['pred'] = preds
        pred_with_actual['actual'] = df.loc[pred_mask, 'rank_actual'].values

        day_corrs = []
        for date, grp in pred_with_actual.groupby('date'):
            if len(grp) >= 20:
                rc, _ = spearmanr(grp['pred'], grp['actual'])
                day_corrs.append(rc)
        avg_rc = np.mean(day_corrs) if day_corrs else 0
        print(f"    {pred_month}: features={len(use_features)}, "
              f"train={len(X_train)}, pred={len(pred_df)}, "
              f"rank_corr={avg_rc:.3f}")

    print(f"[+] ML predicted {total_predicted} rows total")
    return df


def build_enhanced_ranks(df):
    """
    Combine ML predictions with DA forecasts and fallback to baseline.
    Priority:
      1. DA forecast rank (when available, highest quality)
      2. ML-predicted rank (when available)
      3. Baseline 3-feature rank (always available)
    """
    print("[*] Building enhanced rank predictions...")

    has_ml = df['rank_ml'].notna()
    has_da_fcst = df['da_fcst1'].notna()

    # Start with baseline
    df['rank_enhanced'] = df['rank_baseline'].copy()

    # Upgrade to ML where available
    if has_ml.any():
        df.loc[has_ml, 'rank_enhanced'] = df.loc[has_ml, 'rank_ml']

    # For hours with DA forecasts: blend ML + DA forecast ranks
    # DA forecast rank is very strong (0.834 spearman), but ML adds value
    if has_da_fcst.any():
        # Where both ML and DA forecast are available, blend them
        both = has_ml & has_da_fcst
        if both.any():
            # Weight DA forecast more heavily since it has higher rank correlation
            df.loc[both, 'rank_enhanced'] = (
                0.55 * df.loc[both, 'rank_da_fcst1'] +
                0.25 * df.loc[both, 'rank_ml'] +
                0.20 * df.loc[both, 'rank_da_fcst2']
            )

        # Where only DA forecast available (no ML)
        da_only = has_da_fcst & (~has_ml)
        if da_only.any():
            df.loc[da_only, 'rank_enhanced'] = (
                0.60 * df.loc[da_only, 'rank_da_fcst1'] +
                0.40 * df.loc[da_only, 'rank_da_fcst2']
            )

    # Count sources
    n_baseline = (~has_ml & ~has_da_fcst).sum()
    n_ml_only = (has_ml & ~has_da_fcst).sum()
    n_da_only = (has_da_fcst & ~has_ml).sum()
    n_both = (has_ml & has_da_fcst).sum()

    print(f"[+] Enhanced rank sources:")
    print(f"    Baseline only: {n_baseline} hours")
    print(f"    ML only:       {n_ml_only} hours")
    print(f"    DA fcst only:  {n_da_only} hours")
    print(f"    ML + DA fcst:  {n_both} hours")

    # Evaluate rank quality by source
    for label, mask in [('Baseline', ~has_ml & ~has_da_fcst),
                        ('ML only', has_ml & ~has_da_fcst),
                        ('DA+ML', has_ml & has_da_fcst),
                        ('Overall', pd.Series(True, index=df.index))]:
        sub = df[mask]
        if len(sub) < 100:
            continue
        corrs = []
        for date, grp in sub.groupby('date'):
            if len(grp) >= 20:
                rc, _ = spearmanr(grp['rank_enhanced'], grp['da_price'])
                corrs.append(rc)
        if corrs:
            print(f"    {label:12s}: mean rank_corr={np.mean(corrs):.3f}, "
                  f"median={np.median(corrs):.3f}, n_days={len(corrs)}")

    return df


# ============================================================
# SIMULATION
# ============================================================

def run_simulation(df):
    """
    Run BESS simulation with multiple strategies:
      - Enhanced: ML ranks + degradation-aware adaptive cycles
      - Enhanced (no deg): ML ranks, 2 cycles always (for comparison)
      - Baseline: 3-feature rank, 2 cycles always (original 192k result)
      - Perfect: actual prices, 2 cycles always
      - Naive: fixed schedule
    """
    print("\n[*] Running persistent SoC simulation...")
    print(f"[*] Degradation cost: {DEGRADATION_COST_PER_MWH} EUR/MWh "
          f"({DEGRADATION_COST_PER_MWH * 2:.0f} EUR per full cycle)")

    day_counts = df.groupby('date').size()
    all_dates = sorted(day_counts.index.tolist())

    print(f"[+] Total dates: {len(all_dates)}")
    print(f"[+] Date range: {min(all_dates).date()} to {max(all_dates).date()}")

    results = []

    # Persistent SoC state for each strategy
    soc_enhanced = 1      # ML + degradation-aware adaptive
    soc_enh_nodeg = 1     # ML + 2 cycles always (no degradation awareness)
    soc_baseline = 1      # Original 3-feature rank
    soc_perfect = 1       # Perfect foresight
    soc_naive = 1         # Fixed schedule

    skipped = 0
    for date in all_dates:
        day = df[df['date'] == date].sort_values('hour')
        n_hours = len(day)

        if n_hours < 20:
            skipped += 1
            continue

        prices = day['da_price'].values[:n_hours]
        enhanced_ranks = day['rank_enhanced'].values[:n_hours]
        baseline_ranks = day['rank_baseline'].values[:n_hours]

        if np.any(np.isnan(prices)):
            skipped += 1
            continue

        enhanced_ranks = np.nan_to_num(enhanced_ranks, nan=12.0)
        baseline_ranks = np.nan_to_num(baseline_ranks, nan=12.0)

        # --- Enhanced + degradation-aware adaptive ---
        act_enh, rev_enh, soc_enh_trace, cycles_used = optimize_bess_adaptive(
            prices, enhanced_ranks,
            start_soc=soc_enhanced, terminal_penalty=20.0,
            degradation_cost=DEGRADATION_COST_PER_MWH
        )
        end_soc_enh = int(soc_enh_trace[-1])
        n_charge_enh = int(np.sum(act_enh > 0.5))
        n_discharge_enh = int(np.sum(act_enh < -0.5))
        deg_cost_enh = DEGRADATION_COST_PER_MWH * (n_charge_enh + n_discharge_enh)

        # --- Enhanced ML ranks, 2 cycles always (no degradation) ---
        act_end, rev_end, soc_end_trace = optimize_bess_dp(
            prices, enhanced_ranks, n_charge=4, n_discharge=4,
            start_soc=soc_enh_nodeg, terminal_penalty=20.0,
            degradation_cost=0.0
        )
        end_soc_end = int(soc_end_trace[-1])
        n_ch_end = int(np.sum(act_end > 0.5))
        n_dis_end = int(np.sum(act_end < -0.5))

        # --- Baseline (original 3-feature rank, no degradation) ---
        act_base, rev_base, soc_base_trace = optimize_bess_dp(
            prices, baseline_ranks, n_charge=4, n_discharge=4,
            start_soc=soc_baseline, terminal_penalty=20.0,
            degradation_cost=0.0
        )
        end_soc_base = int(soc_base_trace[-1])

        # --- Perfect foresight ---
        act_pf, rev_pf, soc_pf_trace = optimize_bess_dp(
            prices, prices, n_charge=4, n_discharge=4,
            start_soc=soc_perfect, terminal_penalty=20.0,
            degradation_cost=0.0
        )
        end_soc_pf = int(soc_pf_trace[-1])

        # --- Naive fixed schedule ---
        act_naive, end_soc_naive_raw = naive_schedule(n_hours, soc_naive)
        soc_n = soc_naive
        feasible = True
        for h in range(n_hours):
            soc_n += int(act_naive[h])
            if soc_n < 0 or soc_n > 2:
                feasible = False
                break
        if feasible:
            rev_naive = -np.sum(act_naive * prices)
            end_soc_naive = soc_n
        else:
            act_naive = np.zeros(n_hours)
            rev_naive = 0.0
            end_soc_naive = soc_naive

        # Determine rank source for this day
        has_da = day['da_fcst1'].notna().any()
        has_ml = day['rank_ml'].notna().any() if 'rank_ml' in day.columns else False

        if has_da and has_ml:
            rank_source = 'ML+DA'
        elif has_ml:
            rank_source = 'ML'
        elif has_da:
            rank_source = 'DA'
        else:
            rank_source = 'Baseline'

        results.append({
            'date': date,
            'n_hours': n_hours,
            'rank_source': rank_source,
            # Enhanced (degradation-aware adaptive)
            'start_soc_enh': soc_enhanced,
            'end_soc_enh': end_soc_enh,
            'rev_enhanced': rev_enh,
            'cycles_used': cycles_used,
            'n_charge_enh': n_charge_enh,
            'n_discharge_enh': n_discharge_enh,
            'deg_cost_enh': deg_cost_enh,
            'rev_enh_net': rev_enh - deg_cost_enh,
            # Enhanced ML ranks, no degradation
            'start_soc_end': soc_enh_nodeg,
            'end_soc_end': end_soc_end,
            'rev_enh_nodeg': rev_end,
            'deg_cost_end': DEGRADATION_COST_PER_MWH * (n_ch_end + n_dis_end),
            'rev_end_net': rev_end - DEGRADATION_COST_PER_MWH * (n_ch_end + n_dis_end),
            # Baseline
            'start_soc_base': soc_baseline,
            'end_soc_base': end_soc_base,
            'rev_baseline': rev_base,
            # Perfect
            'start_soc_pf': soc_perfect,
            'end_soc_pf': end_soc_pf,
            'rev_perfect': rev_pf,
            # Naive
            'start_soc_naive': soc_naive,
            'end_soc_naive': end_soc_naive,
            'rev_naive': rev_naive,
            # Metrics
            'capture_enhanced': rev_enh / rev_pf if rev_pf > 0 else np.nan,
            'capture_enh_nodeg': rev_end / rev_pf if rev_pf > 0 else np.nan,
            'capture_baseline': rev_base / rev_pf if rev_pf > 0 else np.nan,
            'price_range': prices.max() - prices.min(),
            'price_mean': prices.mean(),
        })

        # Carry SoC
        soc_enhanced = end_soc_enh
        soc_enh_nodeg = end_soc_end
        soc_baseline = end_soc_base
        soc_perfect = end_soc_pf
        soc_naive = end_soc_naive

    if skipped > 0:
        print(f"[-] Skipped {skipped} days (insufficient data)")

    res = pd.DataFrame(results)
    res['date'] = pd.to_datetime(res['date'])
    res = res.sort_values('date').reset_index(drop=True)
    res['year'] = res['date'].dt.year
    res['month'] = res['date'].dt.to_period('M')

    return res


# ============================================================
# REPORTING
# ============================================================

def print_results(res):
    """Print comprehensive comparison table."""
    print("\n" + "=" * 90)
    print("  ENHANCED BESS STRATEGY - RESULTS COMPARISON")
    print("=" * 90)

    print(f"\n  Days simulated: {len(res)}")
    print(f"  Period: {res['date'].min().date()} to {res['date'].max().date()}")
    print(f"  Degradation cost: {DEGRADATION_COST_PER_MWH} EUR/MWh "
          f"({DEGRADATION_COST_PER_MWH * 2:.0f} EUR per full cycle)")

    tot_enh = res['rev_enhanced'].sum()
    tot_enh_net = res['rev_enh_net'].sum()
    tot_end = res['rev_enh_nodeg'].sum()
    tot_end_net = res['rev_end_net'].sum()
    tot_base = res['rev_baseline'].sum()
    tot_pf = res['rev_perfect'].sum()
    tot_naive = res['rev_naive'].sum()
    tot_deg_enh = res['deg_cost_enh'].sum()
    tot_deg_end = res['deg_cost_end'].sum()

    # For baseline/perfect/naive, estimate degradation assuming 2 cycles/day
    # ~4 charge + 4 discharge = 8 MWh throughput per day
    base_deg_est = DEGRADATION_COST_PER_MWH * 8 * len(res)
    pf_deg_est = base_deg_est  # same cycle count

    avg_cap_enh = res['capture_enhanced'].dropna().mean()
    avg_cap_end = res['capture_enh_nodeg'].dropna().mean()
    avg_cap_base = res['capture_baseline'].dropna().mean()

    imp_gross = (tot_enh - tot_base) / tot_base * 100 if tot_base > 0 else 0
    imp_net = (tot_enh_net - (tot_base - base_deg_est)) / (tot_base - base_deg_est) * 100 if (tot_base - base_deg_est) > 0 else 0

    print(f"\n{'--- GROSS REVENUE (before degradation) ---':^90s}")
    print(f"  {'Strategy':<35s} {'Gross (EUR)':>12s} {'Capture':>10s} {'vs Base':>10s}")
    print(f"  {'-'*35} {'-'*12} {'-'*10} {'-'*10}")
    print(f"  {'Enhanced (ML+deg-aware adaptive)':<35s} {tot_enh:>10,.0f}   {avg_cap_enh:>8.1%}   {imp_gross:>+8.1f}%")
    print(f"  {'Enhanced ML (2-cycle always)':<35s} {tot_end:>10,.0f}   {avg_cap_end:>8.1%}   {(tot_end-tot_base)/tot_base*100:>+8.1f}%")
    print(f"  {'Baseline (3-feat rank)':<35s} {tot_base:>10,.0f}   {avg_cap_base:>8.1%}   {'---':>9s}")
    print(f"  {'Perfect Foresight':<35s} {tot_pf:>10,.0f}   {'100.0%':>8s}   {'bound':>9s}")
    print(f"  {'Naive Fixed Schedule':<35s} {tot_naive:>10,.0f}   {'---':>8s}   {'---':>9s}")

    print(f"\n{'--- NET REVENUE (after degradation) ---':^90s}")
    print(f"  {'Strategy':<35s} {'Gross':>10s} {'Deg Cost':>10s} {'Net (EUR)':>12s} {'vs Base Net':>12s}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*12} {'-'*12}")
    print(f"  {'Enhanced (deg-aware adaptive)':<35s} {tot_enh:>8,.0f}   {tot_deg_enh:>8,.0f}   {tot_enh_net:>10,.0f}   {tot_enh_net-(tot_base-base_deg_est):>+10,.0f}")
    print(f"  {'Enhanced ML (2-cycle always)':<35s} {tot_end:>8,.0f}   {tot_deg_end:>8,.0f}   {tot_end_net:>10,.0f}   {tot_end_net-(tot_base-base_deg_est):>+10,.0f}")
    print(f"  {'Baseline (2-cycle always)':<35s} {tot_base:>8,.0f}   {base_deg_est:>8,.0f}   {tot_base-base_deg_est:>10,.0f}   {'---':>11s}")
    print(f"  {'Perfect (2-cycle always)':<35s} {tot_pf:>8,.0f}   {pf_deg_est:>8,.0f}   {tot_pf-pf_deg_est:>10,.0f}   {'bound':>11s}")

    print(f"\n  [*] Key insight: Degradation-aware mode saves {tot_deg_end - tot_deg_enh:,.0f} EUR in battery wear")
    print(f"  [*] Net improvement over baseline: {tot_enh_net - (tot_base - base_deg_est):+,.0f} EUR")

    # Cycle statistics
    print(f"\n{'--- CYCLE MODE DISTRIBUTION ---':^90s}")
    for nc in sorted(res['cycles_used'].unique()):
        n = (res['cycles_used'] == nc).sum()
        sub = res[res['cycles_used'] == nc]
        avg_rev = sub['rev_enhanced'].mean()
        avg_net = sub['rev_enh_net'].mean()
        print(f"  {nc}-cycle days: {n:4d} ({n/len(res)*100:.1f}%)  "
              f"avg gross={avg_rev:.1f} EUR  avg net={avg_net:.1f} EUR")

    # By year
    print(f"\n{'--- BY YEAR (GROSS) ---':^90s}")
    print(f"  {'Year':<6s} {'Enhanced':>12s} {'Enh ML 2c':>12s} {'Baseline':>12s} {'Perfect':>12s} {'Enh Cap':>10s} {'Days':>6s}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*6}")
    for yr in sorted(res['year'].unique()):
        sub = res[res['year'] == yr]
        r_enh = sub['rev_enhanced'].sum()
        r_end = sub['rev_enh_nodeg'].sum()
        r_base = sub['rev_baseline'].sum()
        r_pf = sub['rev_perfect'].sum()
        cap_enh = sub['capture_enhanced'].dropna().mean()
        print(f"  {yr:<6d} {r_enh:>10,.0f}   {r_end:>10,.0f}   {r_base:>10,.0f}   "
              f"{r_pf:>10,.0f}   {cap_enh:>8.1%}   {len(sub):>4d}")

    # By rank source
    print(f"\n{'--- BY RANK SOURCE ---':^90s}")
    print(f"  {'Source':<12s} {'Enhanced':>12s} {'Enh ML 2c':>12s} {'Baseline':>12s} {'Enh Cap':>10s} {'Base Cap':>10s} {'Days':>6s}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*6}")
    for src in ['Baseline', 'ML', 'DA', 'ML+DA']:
        sub = res[res['rank_source'] == src]
        if len(sub) == 0:
            continue
        r_enh = sub['rev_enhanced'].sum()
        r_end = sub['rev_enh_nodeg'].sum()
        r_base = sub['rev_baseline'].sum()
        cap_enh = sub['capture_enhanced'].dropna().mean()
        cap_base = sub['capture_baseline'].dropna().mean()
        print(f"  {src:<12s} {r_enh:>10,.0f}   {r_end:>10,.0f}   {r_base:>10,.0f}   "
              f"{cap_enh:>8.1%}   {cap_base:>8.1%}   {len(sub):>4d}")

    # Monthly
    print(f"\n{'--- MONTHLY ---':^90s}")
    monthly = res.groupby('month').agg(
        rev_enh=('rev_enhanced', 'sum'),
        rev_end=('rev_enh_nodeg', 'sum'),
        rev_base=('rev_baseline', 'sum'),
        rev_pf=('rev_perfect', 'sum'),
        rev_naive=('rev_naive', 'sum'),
        rev_enh_net=('rev_enh_net', 'sum'),
        deg_cost=('deg_cost_enh', 'sum'),
        cap_enh=('capture_enhanced', 'mean'),
        cap_base=('capture_baseline', 'mean'),
        n=('date', 'count'),
    ).reset_index()
    for _, row in monthly.iterrows():
        diff = row['rev_enh'] - row['rev_base']
        sign = '+' if diff >= 0 else ''
        print(f"  {str(row['month']):>8s}: Enh={row['rev_enh']:>8,.0f}  "
              f"ML2c={row['rev_end']:>8,.0f}  Base={row['rev_base']:>8,.0f}  "
              f"Pf={row['rev_pf']:>8,.0f}  Deg={row['deg_cost']:>5,.0f}  "
              f"Cap={row['cap_enh']:.1%}  Diff={sign}{diff:,.0f}  n={row['n']:.0f}")

    # SoC persistence verification
    print(f"\n{'--- SOC PERSISTENCE VERIFICATION ---':^90s}")
    soc_breaks = 0
    for i in range(1, len(res)):
        if res.iloc[i]['start_soc_enh'] != res.iloc[i-1]['end_soc_enh']:
            soc_breaks += 1
    if soc_breaks == 0:
        print(f"  [+] Enhanced SoC chain verified: no breaks in {len(res)} days")
    else:
        print(f"  [!] Enhanced SoC breaks: {soc_breaks}")

    # End-of-day SoC distribution
    print(f"\n{'--- END-OF-DAY SOC DISTRIBUTION ---':^90s}")
    for soc_val in [0, 1, 2]:
        n_enh = (res['end_soc_enh'] == soc_val).sum()
        n_base = (res['end_soc_base'] == soc_val).sum()
        print(f"  SoC={soc_val}: Enhanced={n_enh:4d} ({n_enh/len(res)*100:.1f}%)  "
              f"Baseline={n_base:4d} ({n_base/len(res)*100:.1f}%)")

    print("\n" + "=" * 90)
    return monthly


# ============================================================
# MAIN
# ============================================================

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    df = load_data()

    # 2. Build features
    df = build_features(df)

    # 3. Train ML rank predictor (walk-forward)
    df = train_lightgbm_walk_forward(df)

    # 4. Build enhanced ranks (blend ML + DA forecasts + baseline)
    df = build_enhanced_ranks(df)

    # 5. Run simulation
    res = run_simulation(df)

    # 6. Print results
    monthly = print_results(res)

    # 7. Save outputs
    res.to_csv(OUT / 'enhanced_bess_daily.csv', index=False)
    print(f"\n[+] Daily results saved: {OUT / 'enhanced_bess_daily.csv'}")

    monthly.to_csv(OUT / 'enhanced_bess_monthly.csv', index=False)
    print(f"[+] Monthly results saved: {OUT / 'enhanced_bess_monthly.csv'}")

    return res


if __name__ == '__main__':
    main()
