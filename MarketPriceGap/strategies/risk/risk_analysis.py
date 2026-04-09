"""
Comprehensive Risk Analysis for BESS and Speculative Trading Strategies
=======================================================================
Computes risk metrics, regime analysis, temporal stability,
portfolio construction, and practical risk limits.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data")
BESS_PATH = BASE / "MarketPriceGap" / "da_indicator" / "data" / "bess_persistent_daily.csv"
SPEC_PATH = BASE / "MarketPriceGap" / "da_indicator" / "data" / "speculative_daily.csv"
PRICE_PATH = BASE / "MarketPriceGap" / "data" / "processed" / "hourly_market_prices.csv"
OUT_DIR = BASE / "MarketPriceGap" / "strategies" / "risk"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ===================================================================
# HELPER FUNCTIONS
# ===================================================================

def max_drawdown(equity_curve):
    """Return max drawdown (positive number) and its duration in days."""
    peak = equity_curve.cummax()
    dd = equity_curve - peak
    max_dd = dd.min()
    if max_dd == 0:
        return 0.0, 0
    # duration: longest stretch below the previous peak
    is_in_dd = dd < 0
    groups = (~is_in_dd).cumsum()
    if is_in_dd.any():
        dd_lengths = is_in_dd.groupby(groups).sum()
        max_dur = int(dd_lengths.max())
    else:
        max_dur = 0
    return abs(max_dd), max_dur


def rolling_sharpe(pnl_series, window=30):
    """Annualized Sharpe over a rolling window."""
    roll_mean = pnl_series.rolling(window).mean()
    roll_std = pnl_series.rolling(window).std()
    return (roll_mean / roll_std) * np.sqrt(365)


def streak_analysis(pnl_series):
    """Longest winning and losing streaks."""
    signs = np.sign(pnl_series)
    # Group consecutive same-sign runs
    groups = (signs != signs.shift()).cumsum()
    streaks = signs.groupby(groups).agg(['first', 'count'])
    win_streaks = streaks[streaks['first'] > 0]['count']
    lose_streaks = streaks[streaks['first'] < 0]['count']
    max_win = int(win_streaks.max()) if len(win_streaks) > 0 else 0
    max_lose = int(lose_streaks.max()) if len(lose_streaks) > 0 else 0
    return max_win, max_lose


def compute_risk_metrics(pnl, name, trading_days_year=365):
    """Compute full set of risk metrics for a P&L series."""
    metrics = {}
    metrics['strategy'] = name
    metrics['n_days'] = len(pnl)
    metrics['total_pnl'] = pnl.sum()
    metrics['mean_daily'] = pnl.mean()
    metrics['std_daily'] = pnl.std()
    metrics['skewness'] = pnl.skew()
    metrics['kurtosis'] = pnl.kurtosis()
    metrics['median_daily'] = pnl.median()

    # Win/loss
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    zeros = pnl[pnl == 0]
    metrics['win_rate'] = len(wins) / (len(wins) + len(losses)) if (len(wins) + len(losses)) > 0 else 0
    metrics['avg_win'] = wins.mean() if len(wins) > 0 else 0
    metrics['avg_loss'] = losses.mean() if len(losses) > 0 else 0
    metrics['avg_win_loss_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else float('inf')
    metrics['profit_factor'] = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else float('inf')

    # VaR & CVaR
    metrics['var_95'] = np.percentile(pnl, 5)
    metrics['var_99'] = np.percentile(pnl, 1)
    metrics['cvar_95'] = pnl[pnl <= np.percentile(pnl, 5)].mean()
    metrics['cvar_99'] = pnl[pnl <= np.percentile(pnl, 1)].mean()

    # Ratios
    if pnl.std() > 0:
        metrics['sharpe'] = (pnl.mean() / pnl.std()) * np.sqrt(trading_days_year)
    else:
        metrics['sharpe'] = 0

    downside = pnl[pnl < 0].std()
    if downside > 0:
        metrics['sortino'] = (pnl.mean() / downside) * np.sqrt(trading_days_year)
    else:
        metrics['sortino'] = float('inf') if pnl.mean() > 0 else 0

    # Drawdown
    equity = pnl.cumsum()
    mdd, mdd_dur = max_drawdown(equity)
    metrics['max_drawdown'] = mdd
    metrics['max_drawdown_duration'] = mdd_dur
    annual_return = pnl.mean() * trading_days_year
    metrics['calmar'] = annual_return / mdd if mdd > 0 else float('inf')

    # Best/worst
    metrics['best_day'] = pnl.max()
    metrics['worst_day'] = pnl.min()

    # Streaks
    max_win_streak, max_lose_streak = streak_analysis(pnl)
    metrics['max_win_streak'] = max_win_streak
    metrics['max_lose_streak'] = max_lose_streak

    return metrics


def weekly_pnl(pnl_series, dates):
    """Aggregate P&L by ISO week."""
    df = pd.DataFrame({'date': dates, 'pnl': pnl_series.values})
    df['week'] = pd.to_datetime(df['date']).dt.isocalendar().week.astype(int)
    df['year'] = pd.to_datetime(df['date']).dt.year
    return df.groupby(['year', 'week'])['pnl'].sum()


def monthly_pnl(pnl_series, dates):
    """Aggregate P&L by month."""
    df = pd.DataFrame({'date': dates, 'pnl': pnl_series.values})
    df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
    return df.groupby('month')['pnl'].sum()


# ===================================================================
# LOAD DATA
# ===================================================================

print("=" * 80)
print("  COMPREHENSIVE RISK ANALYSIS - BESS & SPECULATIVE STRATEGIES")
print("=" * 80)
print()

# BESS data
bess = pd.read_csv(BESS_PATH)
bess['date'] = pd.to_datetime(bess['date'])
bess['pnl'] = bess['rev_indicator']  # daily revenue = daily P&L for BESS

# Speculative data
spec = pd.read_csv(SPEC_PATH)
spec['date'] = pd.to_datetime(spec['date'])

# Price data for regime analysis
prices = pd.read_csv(PRICE_PATH, parse_dates=['timestamp_hour'])

print("[+] Loaded BESS: %d days (%s to %s)" % (len(bess), bess['date'].min().date(), bess['date'].max().date()))
print("[+] Loaded Speculative: %d days (%s to %s)" % (len(spec), spec['date'].min().date(), spec['date'].max().date()))
print("[+] Loaded hourly prices: %d rows" % len(prices))
print()

# ===================================================================
# 1. INDIVIDUAL STRATEGY RISK METRICS
# ===================================================================

print("=" * 80)
print("  1. INDIVIDUAL STRATEGY RISK METRICS")
print("=" * 80)
print()

bess_metrics = compute_risk_metrics(bess['pnl'], 'BESS_Indicator', trading_days_year=365)
spec_metrics = compute_risk_metrics(spec['pnl'], 'Speculative', trading_days_year=len(spec) / ((spec['date'].max() - spec['date'].min()).days / 365))

# Weekly and monthly extremes for BESS
bess_weekly = weekly_pnl(bess['pnl'], bess['date'])
bess_monthly = monthly_pnl(bess['pnl'], bess['date'])
bess_metrics['worst_week'] = bess_weekly.min()
bess_metrics['best_week'] = bess_weekly.max()
bess_metrics['worst_month'] = bess_monthly.min()
bess_metrics['best_month'] = bess_monthly.max()

# Weekly and monthly extremes for Speculative
spec_weekly = weekly_pnl(spec['pnl'], spec['date'])
spec_monthly = monthly_pnl(spec['pnl'], spec['date'])
spec_metrics['worst_week'] = spec_weekly.min()
spec_metrics['best_week'] = spec_weekly.max()
spec_metrics['worst_month'] = spec_monthly.min()
spec_metrics['best_month'] = spec_monthly.max()

# Print tables
def print_metric_table(m1, m2):
    fmt_rows = [
        ('Total P&L (EUR)',           'total_pnl',       '%.0f'),
        ('Trading Days',              'n_days',          '%d'),
        ('Mean Daily P&L (EUR)',      'mean_daily',      '%.1f'),
        ('Median Daily P&L (EUR)',    'median_daily',    '%.1f'),
        ('Std Daily P&L (EUR)',       'std_daily',       '%.1f'),
        ('Skewness',                  'skewness',        '%.3f'),
        ('Kurtosis',                  'kurtosis',        '%.3f'),
        ('',                          None,              ''),
        ('Win Rate',                  'win_rate',        '%.1f%%'),
        ('Avg Win (EUR)',             'avg_win',         '%.1f'),
        ('Avg Loss (EUR)',            'avg_loss',        '%.1f'),
        ('Avg Win/Loss Ratio',        'avg_win_loss_ratio', '%.2f'),
        ('Profit Factor',            'profit_factor',   '%.2f'),
        ('',                          None,              ''),
        ('VaR 95% (EUR)',            'var_95',          '%.1f'),
        ('VaR 99% (EUR)',            'var_99',          '%.1f'),
        ('CVaR 95% (EUR)',           'cvar_95',         '%.1f'),
        ('CVaR 99% (EUR)',           'cvar_99',         '%.1f'),
        ('',                          None,              ''),
        ('Sharpe Ratio (ann.)',       'sharpe',          '%.2f'),
        ('Sortino Ratio (ann.)',      'sortino',         '%.2f'),
        ('Calmar Ratio',             'calmar',          '%.2f'),
        ('',                          None,              ''),
        ('Max Drawdown (EUR)',        'max_drawdown',    '%.0f'),
        ('Max DD Duration (days)',    'max_drawdown_duration', '%d'),
        ('Best Day (EUR)',            'best_day',        '%.0f'),
        ('Worst Day (EUR)',           'worst_day',       '%.0f'),
        ('Best Week (EUR)',           'best_week',       '%.0f'),
        ('Worst Week (EUR)',          'worst_week',      '%.0f'),
        ('Best Month (EUR)',          'best_month',      '%.0f'),
        ('Worst Month (EUR)',         'worst_month',     '%.0f'),
        ('',                          None,              ''),
        ('Max Winning Streak (days)', 'max_win_streak',  '%d'),
        ('Max Losing Streak (days)',  'max_lose_streak', '%d'),
    ]

    header = "%-28s  %16s  %16s" % ('Metric', 'BESS Indicator', 'Speculative')
    print(header)
    print("-" * len(header))
    for label, key, fmt in fmt_rows:
        if key is None:
            print()
            continue
        v1 = m1.get(key, 0)
        v2 = m2.get(key, 0)
        if '%' in fmt and 'f%%' in fmt:
            s1 = ("%.1f%%" % (v1 * 100)) if isinstance(v1, float) else str(v1)
            s2 = ("%.1f%%" % (v2 * 100)) if isinstance(v2, float) else str(v2)
        else:
            try:
                s1 = fmt % v1
            except (TypeError, ValueError):
                s1 = str(v1)
            try:
                s2 = fmt % v2
            except (TypeError, ValueError):
                s2 = str(v2)
        print("%-28s  %16s  %16s" % (label, s1, s2))

print_metric_table(bess_metrics, spec_metrics)
print()

# ===================================================================
# 2. REGIME ANALYSIS
# ===================================================================

print("=" * 80)
print("  2. REGIME ANALYSIS")
print("=" * 80)
print()

# Build daily regime features from hourly prices
daily_price = prices.groupby('date').agg(
    price_mean=('da_price', 'mean'),
    price_std=('da_price', 'std'),
    price_min=('da_price', 'min'),
    price_max=('da_price', 'max'),
    price_range=('da_price', lambda x: x.max() - x.min()),
).reset_index()
daily_price['date'] = pd.to_datetime(daily_price['date'])
daily_price['dayofweek'] = daily_price['date'].dt.dayofweek
daily_price['month'] = daily_price['date'].dt.month
daily_price['is_weekend'] = daily_price['dayofweek'] >= 5

# Volatility quartile
vol_75 = daily_price['price_std'].quantile(0.75)
daily_price['high_vol'] = daily_price['price_std'] > vol_75

# Season
def get_season(m):
    if m in [12, 1, 2]:
        return 'Winter'
    elif m in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Shoulder'
daily_price['season'] = daily_price['month'].apply(get_season)

# Extreme price days (any hour < 0 or > 300)
extreme_days = prices[
    (prices['da_price'] < 0) | (prices['da_price'] > 300)
]['date'].unique()
daily_price['extreme_price'] = daily_price['date'].isin(pd.to_datetime(extreme_days))

# Merge regimes into BESS
bess_regime = bess.merge(daily_price[['date', 'is_weekend', 'high_vol', 'season', 'extreme_price', 'price_std']], on='date', how='left', suffixes=('', '_dp'))
# Merge into speculative
spec_regime = spec.merge(daily_price[['date', 'is_weekend', 'high_vol', 'season', 'extreme_price', 'price_std']], on='date', how='left')


def regime_table(df, pnl_col, regime_col, regime_labels=None):
    """Print regime performance table."""
    groups = df.groupby(regime_col)[pnl_col]
    rows = []
    for name, grp in groups:
        label = name if regime_labels is None else regime_labels.get(name, str(name))
        n = len(grp)
        mean_pnl = grp.mean()
        total_pnl = grp.sum()
        std_pnl = grp.std()
        win_r = (grp > 0).sum() / n if n > 0 else 0
        sh = (mean_pnl / std_pnl * np.sqrt(365)) if std_pnl > 0 else 0
        rows.append((label, n, total_pnl, mean_pnl, std_pnl, win_r, sh))
    header = "%-18s  %6s  %10s  %10s  %10s  %8s  %8s" % ('Regime', 'Days', 'Total', 'Mean/d', 'Std/d', 'WinRate', 'Sharpe')
    print(header)
    print("-" * len(header))
    for label, n, total, mean, std, wr, sh in rows:
        print("%-18s  %6d  %10.0f  %10.1f  %10.1f  %7.1f%%  %8.2f" % (label, n, total, mean, std, wr * 100, sh))
    print()


# BESS regime analysis
print("[*] BESS - Volatility Regime:")
regime_table(bess_regime.dropna(subset=['high_vol']), 'pnl', 'high_vol', {True: 'High Vol', False: 'Normal Vol'})

print("[*] BESS - Weekend vs Weekday:")
regime_table(bess_regime.dropna(subset=['is_weekend']), 'pnl', 'is_weekend', {True: 'Weekend', False: 'Weekday'})

print("[*] BESS - Seasonal:")
regime_table(bess_regime.dropna(subset=['season']), 'pnl', 'season')

print("[*] BESS - Extreme Price Days:")
regime_table(bess_regime.dropna(subset=['extreme_price']), 'pnl', 'extreme_price', {True: 'Extreme', False: 'Normal'})

# Speculative regime analysis
print("[*] Speculative - Volatility Regime:")
regime_table(spec_regime.dropna(subset=['high_vol']), 'pnl', 'high_vol', {True: 'High Vol', False: 'Normal Vol'})

print("[*] Speculative - Weekend vs Weekday:")
regime_table(spec_regime.dropna(subset=['is_weekend']), 'pnl', 'is_weekend', {True: 'Weekend', False: 'Weekday'})

print("[*] Speculative - Seasonal:")
regime_table(spec_regime.dropna(subset=['season']), 'pnl', 'season')

print("[*] Speculative - Extreme Price Days:")
regime_table(spec_regime.dropna(subset=['extreme_price']), 'pnl', 'extreme_price', {True: 'Extreme', False: 'Normal'})


# ===================================================================
# 3. TEMPORAL STABILITY
# ===================================================================

print("=" * 80)
print("  3. TEMPORAL STABILITY")
print("=" * 80)
print()

# BESS rolling Sharpe
bess_rolling_sh = rolling_sharpe(bess['pnl'], window=30)
bess_rolling_90 = bess['pnl'].rolling(90).sum()

# Split into halves for stability check
bess_half = len(bess) // 2
bess_h1 = compute_risk_metrics(bess['pnl'].iloc[:bess_half], 'BESS_H1')
bess_h2 = compute_risk_metrics(bess['pnl'].iloc[bess_half:], 'BESS_H2')

spec_half = len(spec) // 2
spec_h1 = compute_risk_metrics(spec['pnl'].iloc[:spec_half], 'Spec_H1')
spec_h2 = compute_risk_metrics(spec['pnl'].iloc[spec_half:], 'Spec_H2')

print("[*] BESS Temporal Stability (split at midpoint):")
print("    First Half:  Mean=%.1f EUR/d, Sharpe=%.2f, MaxDD=%.0f EUR" % (bess_h1['mean_daily'], bess_h1['sharpe'], bess_h1['max_drawdown']))
print("    Second Half: Mean=%.1f EUR/d, Sharpe=%.2f, MaxDD=%.0f EUR" % (bess_h2['mean_daily'], bess_h2['sharpe'], bess_h2['max_drawdown']))
ratio_bess = bess_h2['mean_daily'] / bess_h1['mean_daily'] if bess_h1['mean_daily'] != 0 else 0
if ratio_bess > 1.2:
    print("    [+] Performance IMPROVING in second half (ratio: %.2f)" % ratio_bess)
elif ratio_bess < 0.8:
    print("    [-] Performance DETERIORATING in second half (ratio: %.2f)" % ratio_bess)
else:
    print("    [+] Performance STABLE across halves (ratio: %.2f)" % ratio_bess)
print()

print("[*] Speculative Temporal Stability (split at midpoint):")
print("    First Half:  Mean=%.1f EUR/d, Sharpe=%.2f, MaxDD=%.0f EUR" % (spec_h1['mean_daily'], spec_h1['sharpe'], spec_h1['max_drawdown']))
print("    Second Half: Mean=%.1f EUR/d, Sharpe=%.2f, MaxDD=%.0f EUR" % (spec_h2['mean_daily'], spec_h2['sharpe'], spec_h2['max_drawdown']))
ratio_spec = spec_h2['mean_daily'] / spec_h1['mean_daily'] if spec_h1['mean_daily'] != 0 else 0
if ratio_spec > 1.2:
    print("    [+] Performance IMPROVING in second half (ratio: %.2f)" % ratio_spec)
elif ratio_spec < 0.8:
    print("    [-] Performance DETERIORATING in second half (ratio: %.2f)" % ratio_spec)
else:
    print("    [+] Performance STABLE across halves (ratio: %.2f)" % ratio_spec)
print()

# Quarterly breakdown
bess['quarter'] = bess['date'].dt.to_period('Q')
bess_quarterly = bess.groupby('quarter')['pnl'].agg(['sum', 'mean', 'std', 'count'])
bess_quarterly['sharpe'] = (bess_quarterly['mean'] / bess_quarterly['std']) * np.sqrt(365)
print("[*] BESS Quarterly Performance:")
print("%-10s  %10s  %10s  %10s  %6s  %8s" % ('Quarter', 'Total', 'Mean/d', 'Std/d', 'Days', 'Sharpe'))
print("-" * 62)
for q, row in bess_quarterly.iterrows():
    print("%-10s  %10.0f  %10.1f  %10.1f  %6d  %8.2f" % (q, row['sum'], row['mean'], row['std'], row['count'], row['sharpe']))
print()

spec['quarter'] = spec['date'].dt.to_period('Q')
spec_quarterly = spec.groupby('quarter')['pnl'].agg(['sum', 'mean', 'std', 'count'])
spec_quarterly['sharpe'] = (spec_quarterly['mean'] / spec_quarterly['std']) * np.sqrt(365)
print("[*] Speculative Quarterly Performance:")
print("%-10s  %10s  %10s  %10s  %6s  %8s" % ('Quarter', 'Total', 'Mean/d', 'Std/d', 'Days', 'Sharpe'))
print("-" * 62)
for q, row in spec_quarterly.iterrows():
    print("%-10s  %10.0f  %10.1f  %10.1f  %6d  %8.2f" % (q, row['sum'], row['mean'], row['std'], row['count'], row['sharpe']))
print()

# Rolling 30-day Sharpe summary
valid_sh = bess_rolling_sh.dropna()
if len(valid_sh) > 0:
    print("[*] BESS Rolling 30d Sharpe: mean=%.2f, min=%.2f, max=%.2f, pct_negative=%.1f%%" % (
        valid_sh.mean(), valid_sh.min(), valid_sh.max(), (valid_sh < 0).mean() * 100))

spec_rolling_sh = rolling_sharpe(spec['pnl'], window=30)
valid_sh_spec = spec_rolling_sh.dropna()
if len(valid_sh_spec) > 0:
    print("[*] Spec Rolling 30d Sharpe: mean=%.2f, min=%.2f, max=%.2f, pct_negative=%.1f%%" % (
        valid_sh_spec.mean(), valid_sh_spec.min(), valid_sh_spec.max(), (valid_sh_spec < 0).mean() * 100))
print()

# ===================================================================
# 4. PORTFOLIO ANALYSIS
# ===================================================================

print("=" * 80)
print("  4. PORTFOLIO ANALYSIS (BESS + Speculative)")
print("=" * 80)
print()

# Merge on overlapping dates
merged = bess[['date', 'pnl']].merge(
    spec[['date', 'pnl']],
    on='date',
    suffixes=('_bess', '_spec'),
    how='inner'
)
print("[+] Overlapping dates: %d days (%s to %s)" % (len(merged), merged['date'].min().date(), merged['date'].max().date()))

# Correlation
corr = merged['pnl_bess'].corr(merged['pnl_spec'])
print("[*] Daily P&L correlation: %.4f" % corr)
print()

# Equal weight portfolio
merged['pnl_combined'] = merged['pnl_bess'] + merged['pnl_spec']
port_metrics = compute_risk_metrics(merged['pnl_combined'], 'Portfolio_EqualWeight')

# Compute individual metrics on overlap period only
bess_overlap = compute_risk_metrics(merged['pnl_bess'], 'BESS_Overlap')
spec_overlap = compute_risk_metrics(merged['pnl_spec'], 'Spec_Overlap')

print("[*] Overlap Period Comparison:")
header = "%-25s  %16s  %16s  %16s" % ('Metric', 'BESS', 'Speculative', 'Combined')
print(header)
print("-" * len(header))

compare_rows = [
    ('Total P&L (EUR)',    'total_pnl',    '%.0f'),
    ('Mean Daily (EUR)',   'mean_daily',   '%.1f'),
    ('Std Daily (EUR)',    'std_daily',    '%.1f'),
    ('Sharpe (ann.)',      'sharpe',       '%.2f'),
    ('Sortino (ann.)',     'sortino',      '%.2f'),
    ('Max Drawdown (EUR)', 'max_drawdown', '%.0f'),
    ('Win Rate',           'win_rate',     '%.1f%%'),
    ('Profit Factor',      'profit_factor','%.2f'),
    ('VaR 95% (EUR)',      'var_95',       '%.1f'),
    ('CVaR 95% (EUR)',     'cvar_95',      '%.1f'),
]

for label, key, fmt in compare_rows:
    v1 = bess_overlap.get(key, 0)
    v2 = spec_overlap.get(key, 0)
    v3 = port_metrics.get(key, 0)
    if '%%' in fmt:
        s1 = "%.1f%%" % (v1 * 100)
        s2 = "%.1f%%" % (v2 * 100)
        s3 = "%.1f%%" % (v3 * 100)
    else:
        s1 = fmt % v1
        s2 = fmt % v2
        s3 = fmt % v3
    print("%-25s  %16s  %16s  %16s" % (label, s1, s2, s3))
print()

# Diversification benefit
if bess_overlap['std_daily'] + spec_overlap['std_daily'] > 0:
    div_ratio = port_metrics['std_daily'] / (bess_overlap['std_daily'] + spec_overlap['std_daily'])
    print("[*] Diversification ratio: %.4f (1.0 = no benefit, lower = better)" % div_ratio)
    print("    Risk reduction from diversification: %.1f%%" % ((1 - div_ratio) * 100))
else:
    print("[-] Cannot compute diversification ratio")
print()

# Optimal allocation via mean-variance (simple grid search)
print("[*] Optimal Allocation (mean-variance grid search):")
best_sharpe = -999
best_w = 0
results_alloc = []
for w_bess in np.arange(0, 1.01, 0.05):
    w_spec = 1 - w_bess
    port_pnl = w_bess * merged['pnl_bess'] + w_spec * merged['pnl_spec']
    m = port_pnl.mean()
    s = port_pnl.std()
    sh = (m / s) * np.sqrt(365) if s > 0 else 0
    results_alloc.append((w_bess, w_spec, m, s, sh, port_pnl.sum()))
    if sh > best_sharpe:
        best_sharpe = sh
        best_w = w_bess

print("    Optimal BESS weight: %.0f%%, Speculative weight: %.0f%%" % (best_w * 100, (1 - best_w) * 100))
print("    Optimal portfolio Sharpe: %.2f" % best_sharpe)
print()

# Print allocation frontier
print("%-12s  %-12s  %10s  %10s  %10s  %12s" % ('BESS Wt', 'Spec Wt', 'Mean/d', 'Std/d', 'Sharpe', 'Total PnL'))
print("-" * 72)
for w_b, w_s, m, s, sh, total in results_alloc:
    if w_b % 0.1 < 0.001 or abs(w_b - best_w) < 0.001:
        marker = " <-- optimal" if abs(w_b - best_w) < 0.001 else ""
        print("%-12.0f%%  %-12.0f%%  %10.1f  %10.1f  %10.2f  %12.0f%s" % (w_b * 100, w_s * 100, m, s, sh, total, marker))
print()


# ===================================================================
# 5. PRACTICAL RISK LIMITS
# ===================================================================

print("=" * 80)
print("  5. PRACTICAL RISK LIMITS & RECOMMENDATIONS")
print("=" * 80)
print()

# BESS recommendations
bess_daily_std = bess_metrics['std_daily']
bess_var95 = bess_metrics['var_95']
bess_mean = bess_metrics['mean_daily']
bess_annual = bess_metrics['total_pnl'] / (len(bess) / 365)

print("[*] BESS INDICATOR STRATEGY:")
print("    Expected Annual Revenue: %.0f EUR" % bess_annual)
print("    Daily Stop-Loss Recommendation: %.0f EUR (2x daily std)" % (2 * bess_daily_std))
print("    Monthly Review Threshold: If rolling 30d Sharpe < 0 for 2 consecutive months")
print("    Pause Trading If: 3 consecutive months of negative P&L")
print("    Capital Requirement: %.0f EUR (3x max drawdown as buffer)" % (3 * bess_metrics['max_drawdown']))
print("    Key Risk Indicator: Daily capture_rate < 0.30 for >5 consecutive days")
print()

# Speculative recommendations
spec_daily_std = spec_metrics['std_daily']
spec_var95 = spec_metrics['var_95']
spec_mean = spec_metrics['mean_daily']
spec_annual = spec_metrics['total_pnl'] / (len(spec) / 365)

print("[*] SPECULATIVE STRATEGY:")
print("    Expected Annual Revenue: %.0f EUR" % spec_annual)
print("    Daily Stop-Loss Recommendation: %.0f EUR (2x daily std)" % (2 * spec_daily_std))
print("    Monthly Review Threshold: If monthly P&L < %.0f EUR (worst month threshold)" % (spec_metrics.get('worst_month', 0) * 0.8))
print("    Pause Trading If: Rolling 30d win rate drops below 45%% for 2+ weeks")
print("    Capital Requirement: %.0f EUR (3x max drawdown as buffer)" % (3 * spec_metrics['max_drawdown']))
print("    Key Risk Indicator: Win rate below 50%% for 30 consecutive days")
print()

# Portfolio recommendations
print("[*] COMBINED PORTFOLIO:")
print("    Recommended Allocation: BESS %.0f%% / Speculative %.0f%%" % (best_w * 100, (1 - best_w) * 100))
print("    Combined Annual Revenue (est): %.0f EUR" % (best_w * bess_annual + (1 - best_w) * spec_annual))
print("    Portfolio Sharpe: %.2f" % best_sharpe)
print("    Max Drawdown (combined): %.0f EUR" % port_metrics['max_drawdown'])
print("    Capital Reserve: %.0f EUR (3x portfolio max DD)" % (3 * port_metrics['max_drawdown']))
print()

# Monitoring checklist
print("[*] LIVE TRADING MONITORING CHECKLIST:")
print("    1. Daily: Check each strategy's P&L vs VaR(95%%) limit of %.0f / %.0f EUR" % (abs(bess_var95), abs(spec_var95)))
print("    2. Weekly: Review rolling win rate and capture rate trends")
print("    3. Monthly: Compare realized Sharpe vs historical (BESS: %.2f, Spec: %.2f)" % (bess_metrics['sharpe'], spec_metrics['sharpe']))
print("    4. Quarterly: Full regime analysis - check for structural breaks")
print("    5. Alert: Trigger if daily loss exceeds 3x VaR(99%%) = %.0f / %.0f EUR" % (3 * abs(bess_metrics['var_99']), 3 * abs(spec_metrics['var_99'])))
print()

# ===================================================================
# SAVE RESULTS
# ===================================================================

print("=" * 80)
print("  SAVING RESULTS")
print("=" * 80)
print()

# Combine all metrics into a DataFrame
all_metrics = []
for m in [bess_metrics, spec_metrics, port_metrics, bess_h1, bess_h2, spec_h1, spec_h2, bess_overlap, spec_overlap]:
    all_metrics.append(m)

metrics_df = pd.DataFrame(all_metrics)

# Reorder columns for clarity
col_order = ['strategy', 'n_days', 'total_pnl', 'mean_daily', 'median_daily', 'std_daily',
             'skewness', 'kurtosis', 'win_rate', 'avg_win', 'avg_loss', 'avg_win_loss_ratio',
             'profit_factor', 'var_95', 'var_99', 'cvar_95', 'cvar_99',
             'sharpe', 'sortino', 'calmar', 'max_drawdown', 'max_drawdown_duration',
             'best_day', 'worst_day', 'max_win_streak', 'max_lose_streak']
col_order = [c for c in col_order if c in metrics_df.columns]
extra_cols = [c for c in metrics_df.columns if c not in col_order]
metrics_df = metrics_df[col_order + extra_cols]

out_csv = OUT_DIR / "risk_metrics.csv"
metrics_df.to_csv(out_csv, index=False, float_format='%.4f')
print("[+] Saved risk metrics to: %s" % out_csv)

# Save regime analysis
regime_rows = []
for strategy_name, df_r, pnl_col in [('BESS', bess_regime, 'pnl'), ('Speculative', spec_regime, 'pnl')]:
    for regime_col, regime_map in [
        ('high_vol', {True: 'HighVol', False: 'NormalVol'}),
        ('is_weekend', {True: 'Weekend', False: 'Weekday'}),
        ('season', None),
        ('extreme_price', {True: 'ExtremePx', False: 'NormalPx'}),
    ]:
        valid = df_r.dropna(subset=[regime_col])
        for val, grp in valid.groupby(regime_col):
            label = val if regime_map is None else regime_map.get(val, str(val))
            pnl = grp[pnl_col]
            regime_rows.append({
                'strategy': strategy_name,
                'regime_type': regime_col,
                'regime_value': label,
                'n_days': len(pnl),
                'total_pnl': pnl.sum(),
                'mean_daily': pnl.mean(),
                'std_daily': pnl.std(),
                'win_rate': (pnl > 0).mean(),
                'sharpe': (pnl.mean() / pnl.std() * np.sqrt(365)) if pnl.std() > 0 else 0,
            })

regime_df = pd.DataFrame(regime_rows)
regime_csv = OUT_DIR / "regime_analysis.csv"
regime_df.to_csv(regime_csv, index=False, float_format='%.4f')
print("[+] Saved regime analysis to: %s" % regime_csv)

# Save allocation frontier
alloc_df = pd.DataFrame(results_alloc, columns=['w_bess', 'w_spec', 'mean_daily', 'std_daily', 'sharpe', 'total_pnl'])
alloc_csv = OUT_DIR / "allocation_frontier.csv"
alloc_df.to_csv(alloc_csv, index=False, float_format='%.4f')
print("[+] Saved allocation frontier to: %s" % alloc_csv)

print()
print("=" * 80)
print("  EXECUTIVE RISK SUMMARY")
print("=" * 80)
print()
print("  BESS Indicator Strategy:")
print("    - Annual revenue ~%.0fk EUR, Sharpe %.2f, Max DD %.0fk EUR" % (bess_annual / 1000, bess_metrics['sharpe'], bess_metrics['max_drawdown'] / 1000))
print("    - Win rate %.1f%%, Profit Factor %.2f" % (bess_metrics['win_rate'] * 100, bess_metrics['profit_factor']))
print("    - VaR(95%%) = %.0f EUR/day, CVaR(95%%) = %.0f EUR/day" % (bess_metrics['var_95'], bess_metrics['cvar_95']))
print("    - Performance %s across time" % ("STABLE" if 0.8 <= ratio_bess <= 1.2 else "SHIFTING"))
print()
print("  Speculative Strategy:")
print("    - Annual revenue ~%.0fk EUR, Sharpe %.2f, Max DD %.0fk EUR" % (spec_annual / 1000, spec_metrics['sharpe'], spec_metrics['max_drawdown'] / 1000))
print("    - Win rate %.1f%%, Profit Factor %.2f" % (spec_metrics['win_rate'] * 100, spec_metrics['profit_factor']))
print("    - VaR(95%%) = %.0f EUR/day, CVaR(95%%) = %.0f EUR/day" % (spec_metrics['var_95'], spec_metrics['cvar_95']))
print("    - Performance %s across time" % ("STABLE" if 0.8 <= ratio_spec <= 1.2 else "SHIFTING"))
print()
print("  Portfolio:")
print("    - Correlation: %.4f" % corr)
print("    - Optimal blend: BESS %.0f%% / Spec %.0f%% -> Sharpe %.2f" % (best_w * 100, (1 - best_w) * 100, best_sharpe))
print("    - Combined max DD: %.0fk EUR" % (port_metrics['max_drawdown'] / 1000))
print("    - Capital requirement: %.0fk EUR (3x max DD)" % (3 * port_metrics['max_drawdown'] / 1000))
print()
print("[+] Risk analysis complete.")
