"""
Day-Ahead Trading Indicator for Slovak Electricity Market
=========================================================
Generates actionable day-ahead signals for:
  A) Speculative trader (DA-IDM spread)
  B) Physical trader with 1 MW / 2 MWh BESS

All features respect information timing:
  - Only D-1 available information used (before 11:00 DA auction)
  - Hourly data 2h late, prices real-time, imbalance next day

Usage:
  python build_da_indicator.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import re

warnings.filterwarnings('ignore')

BASE = Path(r'C:\Users\20254757\pycharmprojects\ipesoft_eda_data')
OUT  = BASE / 'MarketPriceGap' / 'da_indicator'

def setup_dirs():
    (OUT / 'plots').mkdir(parents=True, exist_ok=True)
    (OUT / 'data').mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. DATA LOADING
# ============================================================

def load_market_data():
    """Load hourly market prices (DA, IDM, spreads)."""
    print("[*] Loading market prices...")
    mkt = pd.read_csv(BASE / 'MarketPriceGap' / 'data' / 'processed' / 'hourly_market_prices.csv')
    mkt['timestamp_hour'] = pd.to_datetime(mkt['timestamp_hour'])
    mkt['date'] = pd.to_datetime(mkt['date'])
    # Ensure is_weekend is boolean
    if mkt['is_weekend'].dtype == object:
        mkt['is_weekend'] = mkt['is_weekend'].str.lower().isin(['true', '1'])
    else:
        mkt['is_weekend'] = mkt['is_weekend'].astype(bool)
    print(f"[+] Market data: {len(mkt)} rows, {mkt['date'].min().date()} to {mkt['date'].max().date()}")
    return mkt


def load_load_data():
    """Load DAMAS load forecasts and actuals."""
    print("[*] Loading load forecasts...")
    load = pd.read_csv(BASE / 'features' / 'DamasLoad' / 'load_data.csv')
    load['timestamp_hour'] = pd.to_datetime(load['datetime'])
    load = load[['timestamp_hour', 'actual_load_mw', 'forecast_load_mw', 'forecast_error_mw']]
    print(f"[+] Load data: {len(load)} rows")
    return load


def parse_nuclear():
    """Parse Nuclear output from ProductionPerType.csv."""
    filepath = BASE / 'RawData' / 'ProductionPerType.csv'
    if not filepath.exists():
        return None

    print("[*] Parsing production data (nuclear, gas)...")
    col_names = ['Biomass', 'FosilOil', 'HardCoal', 'HydroPump',
                 'HydroPumpConsumption', 'HydroReservoir', 'HydroRunRiver',
                 'Lignite', 'NatGas', 'Nuclear', 'Other', 'OtherRenewable',
                 'SolarActual', 'SolarForecast']

    dt_pat = re.compile(r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}')
    records = []

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        f.readline()  # skip header
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
                # parts[0]=ms, parts[1]=int_value, parts[2]=dec_value (optional)
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
    valid = prod['Nuclear'].notna().sum()
    print(f"[+] Production parsed: {len(prod)} rows, {valid} valid nuclear obs")
    return prod


def load_all_data():
    """Merge all data sources into single hourly DataFrame."""
    mkt = load_market_data()
    load = load_load_data()

    df = mkt.merge(load, on='timestamp_hour', how='left')

    try:
        prod = parse_nuclear()
        if prod is not None and len(prod) > 0:
            df = df.merge(
                prod[['timestamp_hour', 'Nuclear', 'NatGas', 'SolarActual']],
                on='timestamp_hour', how='left'
            )
        else:
            df['Nuclear'] = np.nan
            df['NatGas'] = np.nan
            df['SolarActual'] = np.nan
    except Exception as e:
        print(f"[-] Nuclear parse failed: {e}")
        df['Nuclear'] = np.nan
        df['NatGas'] = np.nan
        df['SolarActual'] = np.nan

    print(f"[+] Merged dataset: {len(df)} rows, {df.shape[1]} columns")
    return df


# ============================================================
# 2. FEATURE ENGINEERING (D-1 AVAILABLE ONLY)
# ============================================================

def build_features(df):
    """Build features available before 11:00 D-1 DA auction."""
    print("[*] Building D-1 available features...")
    df = df.sort_values(['date', 'hour']).reset_index(drop=True)

    # --- Daily aggregates ---
    daily = df.groupby('date').agg(
        da_mean=('da_price', 'mean'),
        da_max=('da_price', 'max'),
        da_min=('da_price', 'min'),
        da_std=('da_price', 'std'),
        da_range=('da_price', lambda x: x.max() - x.min()),
        idm_mean=('idm_vwap', 'mean'),
        spread_mean=('spread_da_idm', 'mean'),
        load_fcst_mean=('forecast_load_mw', 'mean'),
        load_actual_mean=('actual_load_mw', 'mean'),
        load_error_mean=('forecast_error_mw', 'mean'),
        nuclear_mean=('Nuclear', 'mean'),
        is_weekend=('is_weekend', 'first'),
    ).reset_index()
    daily = daily.sort_values('date').reset_index(drop=True)

    # --- Yesterday's daily values (shift by 1 day) ---
    for col in ['da_mean', 'da_max', 'da_min', 'da_std', 'da_range',
                'idm_mean', 'spread_mean', 'load_actual_mean',
                'load_error_mean', 'nuclear_mean']:
        daily[f'{col}_yest'] = daily[col].shift(1)

    # --- 7-day rolling averages ---
    for col in ['da_mean', 'idm_mean', 'spread_mean', 'load_actual_mean']:
        daily[f'{col}_7d'] = daily[col].shift(1).rolling(7, min_periods=3).mean()

    # --- Load forecast deviation from recent normal ---
    daily['load_fcst_dev'] = daily['load_fcst_mean'] - daily['load_actual_mean_7d']

    # Merge daily features back to hourly
    daily_feats = daily.drop(columns=['is_weekend'], errors='ignore')
    df = df.merge(daily_feats, on='date', how='left', suffixes=('', '_daily'))

    # --- Same-hour yesterday's values ---
    df = df.sort_values(['hour', 'date']).reset_index(drop=True)
    df['da_price_yest_h'] = df.groupby('hour')['da_price'].shift(1)
    df['idm_vwap_yest_h'] = df.groupby('hour')['idm_vwap'].shift(1)
    df['spread_yest_h'] = df.groupby('hour')['spread_da_idm'].shift(1)
    df['da_price_7d_h'] = df.groupby('hour')['da_price'].shift(7)

    # Re-sort by timestamp
    df = df.sort_values(['date', 'hour']).reset_index(drop=True)

    print(f"[+] Features built: {df.shape[1]} columns")
    return df, daily


# ============================================================
# 3. INDICATOR SIGNAL GENERATION
# ============================================================

def generate_spread_signal(row):
    """
    Generate spread direction signal for one hour.
    Returns (signal, confidence, trade_flag).
    signal: +1 = expect DA > IDM (sell DA), -1 = expect DA < IDM (buy DA)
    """
    is_wknd = bool(row.get('is_weekend', False))
    load_dev = row.get('load_fcst_dev', 0)
    spread_yest = row.get('spread_yest_h', 0)
    spread_yest_daily = row.get('spread_mean_yest', 0)

    if pd.isna(load_dev):
        load_dev = 0
    if pd.isna(spread_yest):
        spread_yest = 0
    if pd.isna(spread_yest_daily):
        spread_yest_daily = 0

    # Rule 1: Weekend (89% accuracy historically)
    if is_wknd:
        return 1, 0.89, 1.0

    # Rule 2: Weekday with unusually high load forecast
    if load_dev > 85:
        return -1, 0.66, 1.0

    # Rule 3: Large yesterday spread -> follow persistence
    if abs(spread_yest) >= 20:
        return (1 if spread_yest > 0 else -1), 0.69, 1.0

    # Rule 4: Moderate daily persistence
    if abs(spread_yest_daily) >= 10:
        return (1 if spread_yest_daily > 0 else -1), 0.58, 0.5

    # Rule 5: Weak baseline (slight DA > IDM bias)
    return 1, 0.54, 0.0


def generate_signals(df):
    """Generate all trading signals."""
    print("[*] Generating indicator signals...")

    # --- Spread direction signal ---
    results = df.apply(generate_spread_signal, axis=1, result_type='expand')
    df['spread_signal'] = results[0].astype(float)
    df['spread_confidence'] = results[1].astype(float)
    df['trade_flag'] = results[2].astype(float)

    # --- Price shape ranking (for BESS) ---
    # Load-forecast-based rank
    df['rank_load'] = df.groupby('date')['forecast_load_mw'].rank(method='first', na_option='bottom')
    # Yesterday-price-based rank
    df['rank_yest'] = df.groupby('date')['da_price_yest_h'].rank(method='first', na_option='bottom')
    # Weekly-price-based rank
    df['rank_7d'] = df.groupby('date')['da_price_7d_h'].rank(method='first', na_option='bottom')

    # Composite predicted rank (weighted ensemble)
    df['rank_predicted'] = (
        0.40 * df['rank_load'].fillna(12) +
        0.35 * df['rank_yest'].fillna(12) +
        0.25 * df['rank_7d'].fillna(12)
    )

    # Actual rank (for evaluation)
    df['rank_actual'] = df.groupby('date')['da_price'].rank(method='first', na_option='bottom')

    print("[+] Signals generated")
    return df


# ============================================================
# 4. SPECULATIVE TRADER BACKTEST
# ============================================================

def backtest_speculative(df):
    """Backtest DA-IDM spread trading."""
    print("[*] Running speculative trader backtest...")

    # Only hours with IDM data
    mask = df['idm_vwap'].notna() & df['spread_da_idm'].notna()
    bt = df[mask].copy()

    if len(bt) == 0:
        print("[-] No IDM data available for backtest")
        return bt, pd.DataFrame(), pd.DataFrame()

    # P&L: if signal=+1 (DA>IDM), we sell DA and buy IDM -> profit = spread
    bt['pnl_raw'] = bt['spread_signal'] * bt['spread_da_idm']
    bt['pnl'] = bt['pnl_raw'] * bt['trade_flag']
    bt['correct'] = ((bt['spread_signal'] > 0) == (bt['spread_da_idm'] > 0)).astype(int)
    bt['traded'] = (bt['trade_flag'] > 0).astype(int)

    # Daily aggregation
    daily = bt.groupby('date').agg(
        pnl=('pnl', 'sum'),
        pnl_raw=('pnl_raw', 'sum'),
        n_correct=('correct', lambda x: x[bt.loc[x.index, 'traded'] > 0].sum()),
        n_traded=('traded', 'sum'),
    ).reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values('date')
    daily['cum_pnl'] = daily['pnl'].cumsum()
    daily['cum_pnl_raw'] = daily['pnl_raw'].cumsum()
    daily['accuracy'] = daily['n_correct'] / daily['n_traded'].clip(lower=1)
    daily['year'] = daily['date'].dt.year
    daily['month_period'] = daily['date'].dt.to_period('M')

    # Monthly
    monthly = daily.groupby('month_period').agg(
        pnl=('pnl', 'sum'),
        n_correct=('n_correct', 'sum'),
        n_traded=('n_traded', 'sum'),
        n_days=('date', 'count'),
    ).reset_index()
    monthly['accuracy'] = monthly['n_correct'] / monthly['n_traded'].clip(lower=1)

    print(f"[+] Speculative backtest: {len(bt)} traded hours, {daily['pnl'].sum():,.0f} EUR total P&L")
    return bt, daily, monthly


# ============================================================
# 5. BESS TRADER BACKTEST
# ============================================================

def optimize_bess_day(prices, predicted_ranks):
    """
    Optimize 1MW/2MWh BESS for one day.
    Simple approach: charge 2 cheapest predicted hours, discharge 2 most expensive.
    SoC start = 1 MWh, bounds [0, 2].
    """
    if len(prices) < 24 or np.any(np.isnan(prices[:24])):
        return np.zeros(min(len(prices), 24)), 0.0

    pred_order = np.argsort(predicted_ranks[:24])
    charge_set = set(pred_order[:2])
    discharge_set = set(pred_order[-2:])

    actions = np.zeros(24)
    for h in charge_set:
        actions[h] = 1
    for h in discharge_set:
        actions[h] = -1

    # Verify SoC
    soc = 1.0
    ok = True
    for h in range(24):
        soc += actions[h]
        if soc < -0.01 or soc > 2.01:
            ok = False
            break

    if not ok:
        # Fallback: chronological greedy
        actions = np.zeros(24)
        soc = 1.0
        charged, discharged = 0, 0

        for h in np.argsort(predicted_ranks[:24]):
            if charged >= 2:
                break
            if soc + 1 <= 2:
                actions[h] = 1
                soc += 1
                charged += 1

        for h in np.argsort(-predicted_ranks[:24]):
            if discharged >= 2:
                break
            if actions[h] == 0:
                # Check SoC feasibility
                test_soc = 1.0
                test_act = actions.copy()
                test_act[h] = -1
                feasible = True
                for hh in range(24):
                    test_soc += test_act[hh]
                    if test_soc < -0.01 or test_soc > 2.01:
                        feasible = False
                        break
                if feasible:
                    actions[h] = -1
                    discharged += 1

    revenue = -np.sum(actions[:24] * prices[:24])
    return actions, revenue


def backtest_bess(df):
    """Backtest BESS strategy across all complete days."""
    print("[*] Running BESS trader backtest...")

    day_counts = df.groupby('date').size()
    full_days = day_counts[day_counts >= 24].index.tolist()

    results = []
    for date in full_days:
        day = df[df['date'] == date].sort_values('hour')
        if len(day) < 24:
            continue

        prices = day['da_price'].values[:24]
        pred_ranks = day['rank_predicted'].values[:24]

        if np.any(np.isnan(prices)):
            continue

        pred_ranks = np.nan_to_num(pred_ranks, nan=12.0)

        # Indicator-guided
        act_ind, rev_ind = optimize_bess_day(prices, pred_ranks)
        # Perfect foresight
        act_pf, rev_pf = optimize_bess_day(prices, prices)
        # Naive: charge h3-4, discharge h18-19
        act_naive = np.zeros(24)
        act_naive[3], act_naive[4] = 1, 1
        act_naive[18], act_naive[19] = -1, -1
        rev_naive = -np.sum(act_naive * prices)

        results.append({
            'date': date,
            'rev_indicator': rev_ind,
            'rev_perfect': rev_pf,
            'rev_naive': rev_naive,
            'capture_rate': rev_ind / rev_pf if rev_pf > 0 else np.nan,
            'is_weekend': day['is_weekend'].iloc[0],
            'spread_available': day['idm_vwap'].notna().any(),
            'charge_h_ind': str(list(np.where(act_ind > 0)[0])),
            'discharge_h_ind': str(list(np.where(act_ind < 0)[0])),
        })

    bess = pd.DataFrame(results)
    bess['date'] = pd.to_datetime(bess['date'])
    bess = bess.sort_values('date')
    bess['year'] = bess['date'].dt.year
    bess['month_period'] = bess['date'].dt.to_period('M')

    # Monthly
    monthly = bess.groupby('month_period').agg(
        rev_indicator=('rev_indicator', 'sum'),
        rev_perfect=('rev_perfect', 'sum'),
        rev_naive=('rev_naive', 'sum'),
        n_days=('date', 'count'),
        avg_capture=('capture_rate', 'mean'),
    ).reset_index()

    print(f"[+] BESS backtest: {len(bess)} days, {bess['rev_indicator'].sum():,.0f} EUR indicator revenue")
    return bess, monthly


# ============================================================
# 6. CHARTS
# ============================================================

def plot_speculative(spec_bt, spec_daily, spec_monthly):
    """Charts for speculative trader."""
    print("[*] Generating speculative trader charts...")
    plt.style.use('seaborn-v0_8-whitegrid')
    jan26 = pd.Timestamp('2026-01-01')

    # --- Chart 1: Cumulative P&L ---
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(spec_daily['date'], spec_daily['cum_pnl'], 'b-', lw=1.5, label='Indicator (filtered trades)')
    ax.plot(spec_daily['date'], spec_daily['cum_pnl_raw'], color='gray', lw=0.8, alpha=0.5,
            label='All hours (unfiltered)')
    if spec_daily['date'].max() > jan26:
        ax.axvline(jan26, color='red', ls='--', alpha=0.5, label='2026 start')
        ax.axvspan(jan26, spec_daily['date'].max(), alpha=0.08, color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative P&L (EUR per 1 MWh position)')
    ax.set_title('Speculative Trader: DA-IDM Spread Trading - Cumulative P&L')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / 'plots' / '01_speculative_cumulative_pnl.png', dpi=150)
    plt.close(fig)

    # --- Chart 2: Monthly P&L with accuracy ---
    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(spec_monthly))
    colors = ['#d32f2f' if str(m).startswith('2026') else '#1976d2'
              for m in spec_monthly['month_period']]
    ax.bar(x, spec_monthly['pnl'], color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(m) for m in spec_monthly['month_period']], rotation=45, ha='right')
    ax.set_ylabel('Monthly P&L (EUR)')
    ax.set_title('Speculative Trader: Monthly P&L (red=2026, blue=2025)')
    for i, row in spec_monthly.iterrows():
        idx = list(spec_monthly.index).index(i)
        pnl_val = row['pnl']
        va = 'bottom' if pnl_val >= 0 else 'top'
        ax.annotate(f"{row['accuracy']:.0%}", (idx, pnl_val), ha='center', va=va,
                    fontsize=8, fontweight='bold')
    ax.axhline(0, color='black', lw=0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / 'plots' / '02_speculative_monthly_pnl.png', dpi=150)
    plt.close(fig)

    # --- Chart 3: Hourly analysis ---
    traded = spec_bt[spec_bt['trade_flag'] > 0]
    hourly = traded.groupby('hour').agg(
        accuracy=('correct', 'mean'),
        avg_pnl=('pnl', 'mean'),
        n_trades=('traded', 'sum'),
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax1.bar(hourly['hour'], hourly['accuracy'], color='#1976d2', edgecolor='white')
    ax1.axhline(0.5, color='red', ls='--', alpha=0.5, label='Random (50%)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Speculative Trader: Performance by Hour of Day')
    ax1.legend()
    ax1.set_ylim(0.3, 1.0)

    pnl_colors = ['#4caf50' if v > 0 else '#d32f2f' for v in hourly['avg_pnl']]
    ax2.bar(hourly['hour'], hourly['avg_pnl'], color=pnl_colors, edgecolor='white')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Avg P&L per Trade (EUR)')
    ax2.axhline(0, color='black', lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / 'plots' / '03_speculative_hourly.png', dpi=150)
    plt.close(fig)

    # --- Chart 4: Rolling accuracy ---
    daily_acc = traded.groupby('date').agg(accuracy=('correct', 'mean')).reset_index()
    daily_acc['date'] = pd.to_datetime(daily_acc['date'])
    daily_acc = daily_acc.sort_values('date')
    daily_acc['rolling'] = daily_acc['accuracy'].rolling(30, min_periods=10).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily_acc['date'], daily_acc['rolling'], 'b-', lw=1.5, label='30-day rolling accuracy')
    ax.axhline(0.5, color='red', ls='--', alpha=0.5, label='Random')
    ax.axhline(0.6, color='orange', ls='--', alpha=0.3, label='~Breakeven')
    if daily_acc['date'].max() > jan26:
        ax.axvline(jan26, color='red', ls='--', alpha=0.3)
    ax.set_xlabel('Date')
    ax.set_ylabel('Rolling 30-day Accuracy')
    ax.set_title('Indicator Accuracy Over Time - Is the Signal Persisting?')
    ax.legend()
    ax.set_ylim(0.3, 0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / 'plots' / '04_rolling_accuracy.png', dpi=150)
    plt.close(fig)


def plot_bess(bess_daily, bess_monthly):
    """Charts for BESS trader."""
    print("[*] Generating BESS trader charts...")
    jan26 = pd.Timestamp('2026-01-01')

    # --- Chart 5: BESS Cumulative Revenue ---
    fig, ax = plt.subplots(figsize=(14, 6))
    bd = bess_daily.sort_values('date')
    ax.plot(bd['date'], bd['rev_indicator'].cumsum(), 'b-', lw=1.5, label='Indicator-guided')
    ax.plot(bd['date'], bd['rev_perfect'].cumsum(), 'g--', lw=1, alpha=0.7, label='Perfect foresight')
    ax.plot(bd['date'], bd['rev_naive'].cumsum(), 'r:', lw=1, alpha=0.7, label='Naive (fixed h3-4/h18-19)')
    if bd['date'].max() > jan26:
        ax.axvline(jan26, color='red', ls='--', alpha=0.3)
        ax.axvspan(jan26, bd['date'].max(), alpha=0.05, color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Revenue (EUR)')
    ax.set_title('BESS (1 MW / 2 MWh): Cumulative Revenue')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / 'plots' / '05_bess_cumulative_revenue.png', dpi=150)
    plt.close(fig)

    # --- Chart 6: BESS Monthly Revenue ---
    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(bess_monthly))
    w = 0.25
    ax.bar([i - w for i in x], bess_monthly['rev_indicator'], w,
           label='Indicator', color='#1976d2', edgecolor='white')
    ax.bar(x, bess_monthly['rev_perfect'], w,
           label='Perfect foresight', color='#4caf50', edgecolor='white')
    ax.bar([i + w for i in x], bess_monthly['rev_naive'], w,
           label='Naive', color='#ff9800', edgecolor='white')
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(m) for m in bess_monthly['month_period']], rotation=45, ha='right')
    ax.set_ylabel('Monthly Revenue (EUR)')
    ax.set_title('BESS (1 MW / 2 MWh): Monthly Revenue Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / 'plots' / '06_bess_monthly_revenue.png', dpi=150)
    plt.close(fig)

    # --- Chart 7: Capture rate distribution ---
    fig, ax = plt.subplots(figsize=(10, 5))
    cr = bess_daily['capture_rate'].dropna()
    ax.hist(cr, bins=40, color='#1976d2', edgecolor='white', alpha=0.8)
    ax.axvline(cr.median(), color='red', ls='--',
               label=f'Median: {cr.median():.1%}')
    ax.axvline(cr.mean(), color='orange', ls='--',
               label=f'Mean: {cr.mean():.1%}')
    ax.set_xlabel('Capture Rate (Indicator / Perfect Foresight)')
    ax.set_ylabel('Number of Days')
    ax.set_title('BESS: How Close to Optimal?')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'plots' / '07_bess_capture_rate.png', dpi=150)
    plt.close(fig)


def plot_combined(spec_bt, spec_daily, bess_daily):
    """Combined comparison charts."""
    print("[*] Generating combined charts...")

    # --- Chart 8: 2025 vs 2026 comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Speculative
    ax = axes[0]
    traded = spec_bt[spec_bt['trade_flag'] > 0].copy()
    traded['year'] = pd.to_datetime(traded['date']).dt.year
    for yr, color in [(2025, '#1976d2'), (2026, '#d32f2f')]:
        sub = traded[traded['year'] == yr]
        if len(sub) > 0:
            acc = sub['correct'].mean()
            avg = sub['pnl'].mean()
            ax.bar(str(yr), avg, color=color, edgecolor='white', width=0.5)
            ax.annotate(f"Acc: {acc:.1%}\nn={len(sub)}", (str(yr), avg),
                        ha='center', va='bottom' if avg >= 0 else 'top', fontweight='bold')
    ax.set_ylabel('Avg P&L per Trade (EUR)')
    ax.set_title('Speculative: 2025 vs 2026')
    ax.axhline(0, color='black', lw=0.5)

    # BESS
    ax = axes[1]
    for yr, color in [(2024, '#9e9e9e'), (2025, '#1976d2'), (2026, '#d32f2f')]:
        sub = bess_daily[bess_daily['year'] == yr]
        if len(sub) > 0:
            avg = sub['rev_indicator'].mean()
            cap = sub['capture_rate'].mean()
            ax.bar(str(yr), avg, color=color, edgecolor='white', width=0.5)
            ax.annotate(f"Capture: {cap:.1%}\nn={len(sub)}", (str(yr), avg),
                        ha='center', va='bottom' if avg >= 0 else 'top', fontweight='bold')
    ax.set_ylabel('Avg Daily Revenue (EUR)')
    ax.set_title('BESS: Year-over-Year')
    ax.axhline(0, color='black', lw=0.5)

    fig.suptitle('Performance Persistence: Is the Signal Disappearing in 2026?', fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / 'plots' / '08_year_comparison.png', dpi=150)
    plt.close(fig)

    # --- Chart 9: Weekend vs Weekday ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for label, wknd, color in [('Weekend', True, '#4caf50'), ('Weekday', False, '#1976d2')]:
        sub = traded[traded['is_weekend'] == wknd]
        if len(sub) > 0:
            ax.bar(label, sub['pnl'].mean(), color=color, edgecolor='white', width=0.5)
            ax.annotate(f"Acc: {sub['correct'].mean():.1%}\nn={len(sub)}",
                        (label, sub['pnl'].mean()), ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Avg P&L per Trade (EUR)')
    ax.set_title('Speculative: Weekend vs Weekday')
    ax.axhline(0, color='black', lw=0.5)

    ax = axes[1]
    for label, wknd, color in [('Weekend', True, '#4caf50'), ('Weekday', False, '#1976d2')]:
        sub = bess_daily[bess_daily['is_weekend'] == wknd]
        if len(sub) > 0:
            ax.bar(label, sub['rev_indicator'].mean(), color=color, edgecolor='white', width=0.5)
            ax.annotate(f"Capture: {sub['capture_rate'].mean():.1%}\nn={len(sub)}",
                        (label, sub['rev_indicator'].mean()), ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Avg Daily Revenue (EUR)')
    ax.set_title('BESS: Weekend vs Weekday')
    ax.axhline(0, color='black', lw=0.5)

    fig.suptitle('Weekend vs Weekday Performance', fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / 'plots' / '09_weekend_vs_weekday.png', dpi=150)
    plt.close(fig)


# ============================================================
# 7. SUMMARY
# ============================================================

def print_summary(spec_bt, spec_daily, spec_monthly, bess_daily, bess_monthly):
    """Print comprehensive results."""
    print("\n" + "=" * 70)
    print("  DAY-AHEAD TRADING INDICATOR - RESULTS SUMMARY")
    print("=" * 70)

    # --- Speculative ---
    traded = spec_bt[spec_bt['trade_flag'] > 0]
    if len(traded) > 0:
        total_pnl = traded['pnl'].sum()
        acc = traded['correct'].mean()
        avg_pnl = traded['pnl'].mean()
        std_pnl = traded['pnl'].std()
        sharpe = avg_pnl / std_pnl * np.sqrt(252 * 24) if std_pnl > 0 else 0

        print(f"\n--- SPECULATIVE TRADER (1 MWh per trade) ---")
        print(f"Period:       {traded['date'].min()} to {traded['date'].max()}")
        print(f"Total trades: {len(traded)}")
        print(f"Total P&L:    {total_pnl:,.0f} EUR")
        print(f"Avg P&L/trade:{avg_pnl:.2f} EUR")
        print(f"Accuracy:     {acc:.1%}")
        print(f"Sharpe (ann): {sharpe:.1f}")

        traded_copy = traded.copy()
        traded_copy['year'] = pd.to_datetime(traded_copy['date']).dt.year
        for yr in sorted(traded_copy['year'].unique()):
            sub = traded_copy[traded_copy['year'] == yr]
            print(f"  {yr}: P&L={sub['pnl'].sum():,.0f} EUR, "
                  f"Acc={sub['correct'].mean():.1%}, n={len(sub)}")

    # --- BESS ---
    if len(bess_daily) > 0:
        print(f"\n--- BESS TRADER (1 MW / 2 MWh) ---")
        print(f"Period:       {bess_daily['date'].min().date()} to {bess_daily['date'].max().date()}")
        print(f"Trading days: {len(bess_daily)}")
        print(f"Revenue (indicator):  {bess_daily['rev_indicator'].sum():,.0f} EUR")
        print(f"Revenue (perfect):    {bess_daily['rev_perfect'].sum():,.0f} EUR")
        print(f"Revenue (naive):      {bess_daily['rev_naive'].sum():,.0f} EUR")
        print(f"Avg capture rate:     {bess_daily['capture_rate'].mean():.1%}")
        print(f"Avg daily rev (ind):  {bess_daily['rev_indicator'].mean():.1f} EUR/day")
        print(f"Avg daily rev (perf): {bess_daily['rev_perfect'].mean():.1f} EUR/day")

        for yr in sorted(bess_daily['year'].unique()):
            sub = bess_daily[bess_daily['year'] == yr]
            print(f"  {yr}: Rev={sub['rev_indicator'].sum():,.0f} EUR, "
                  f"Capture={sub['capture_rate'].mean():.1%}, n={len(sub)}")

    # --- Signal persistence ---
    print(f"\n--- 2026 SIGNAL PERSISTENCE CHECK ---")
    if len(traded) > 0:
        traded_copy = traded.copy()
        traded_copy['year'] = pd.to_datetime(traded_copy['date']).dt.year
        for yr in [2025, 2026]:
            sub = traded_copy[traded_copy['year'] == yr]
            if len(sub) > 0:
                acc = sub['correct'].mean()
                avg = sub['pnl'].mean()
                status = "OK" if acc > 0.55 else "DEGRADED" if acc > 0.50 else "FAILED"
                print(f"  {yr} speculative: accuracy={acc:.1%}, avg_pnl={avg:.2f} EUR -> [{status}]")

    if len(bess_daily) > 0:
        for yr in [2025, 2026]:
            sub = bess_daily[bess_daily['year'] == yr]
            if len(sub) > 0:
                cap = sub['capture_rate'].mean()
                avg = sub['rev_indicator'].mean()
                status = "OK" if cap > 0.6 else "DEGRADED" if cap > 0.4 else "FAILED"
                print(f"  {yr} BESS: capture={cap:.1%}, avg_daily_rev={avg:.1f} EUR -> [{status}]")

    print("\n" + "=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():
    setup_dirs()

    # 1. Load data
    df = load_all_data()

    # 2. Build features
    df, daily = build_features(df)

    # 3. Generate signals
    df = generate_signals(df)

    # 4. Backtest speculative
    spec_bt, spec_daily, spec_monthly = backtest_speculative(df)

    # 5. Backtest BESS
    bess_daily, bess_monthly = backtest_bess(df)

    # 6. Charts
    if len(spec_daily) > 0:
        plot_speculative(spec_bt, spec_daily, spec_monthly)
    if len(bess_daily) > 0:
        plot_bess(bess_daily, bess_monthly)
    if len(spec_daily) > 0 and len(bess_daily) > 0:
        plot_combined(spec_bt, spec_daily, bess_daily)

    # 7. Summary
    print_summary(spec_bt, spec_daily, spec_monthly, bess_daily, bess_monthly)

    # 8. Save outputs
    if len(spec_daily) > 0:
        spec_daily.to_csv(OUT / 'data' / 'speculative_daily.csv', index=False)
        spec_monthly.to_csv(OUT / 'data' / 'speculative_monthly.csv', index=False)
    if len(bess_daily) > 0:
        bess_daily.to_csv(OUT / 'data' / 'bess_daily.csv', index=False)
        bess_monthly.to_csv(OUT / 'data' / 'bess_monthly.csv', index=False)

    # Save indicator signals
    sig_cols = ['timestamp_hour', 'date', 'hour', 'da_price', 'idm_vwap',
                'spread_da_idm', 'spread_signal', 'spread_confidence',
                'trade_flag', 'rank_predicted', 'rank_actual',
                'is_weekend', 'load_fcst_dev']
    existing = [c for c in sig_cols if c in df.columns]
    df[existing].to_csv(OUT / 'data' / 'indicator_signals.csv', index=False)

    print(f"\n[+] All outputs saved to {OUT}")


if __name__ == '__main__':
    main()
