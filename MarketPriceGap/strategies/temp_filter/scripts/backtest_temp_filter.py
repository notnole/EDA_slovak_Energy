"""
Temperature-filtered "Sell DA, Buy Imbalance" backtest.

Strategy: each hour, sell 1 MWh at DA price, buy back at imbalance settlement price.
P&L per hour = da_price - imb_settlement_price = spread_da_imb
Daily P&L = sum of 24 hourly spreads.

Temperature filter: skip trading on days with extreme temperatures.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PATHS
# ============================================================================
BASE = r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data"
MARKET_CSV = f"{BASE}/MarketPriceGap/data/processed/hourly_market_prices.csv"
ACTUAL_TEMP_CSV = f"{BASE}/RawData/ActualTemp2024-2026.csv"
GFS_TEMP_CSV = f"{BASE}/RawData/Tampretures.csv"
OUT_PNG = f"{BASE}/MarketPriceGap/strategies/temp_filter/temp_filter_backtest.png"


# ============================================================================
# STEP 1: Parse and merge data
# ============================================================================
print("=" * 70)
print("[*] STEP 1: Loading and parsing data")
print("=" * 70)

# --- 1a. Market data ---
mkt = pd.read_csv(MARKET_CSV, parse_dates=['timestamp_hour'])
mkt = mkt[['timestamp_hour', 'da_price', 'imb_settlement_price', 'spread_da_imb']].copy()
mkt = mkt.dropna(subset=['spread_da_imb'])
mkt['date'] = mkt['timestamp_hour'].dt.date
print(f"[+] Market data: {len(mkt)} hourly rows, {mkt['date'].nunique()} days")
print(f"    Period: {mkt['timestamp_hour'].min()} to {mkt['timestamp_hour'].max()}")

# Daily P&L
daily_pnl = mkt.groupby('date')['spread_da_imb'].sum().reset_index()
daily_pnl.columns = ['date', 'daily_pnl']
daily_pnl['date'] = pd.to_datetime(daily_pnl['date'])
print(f"[+] Daily P&L: {len(daily_pnl)} days, total = {daily_pnl['daily_pnl'].sum():.1f} EUR")


# --- 1b. Actual temperature (full period) ---
def parse_actual_temp(filepath):
    """Parse ActualTemp2024-2026.csv with European decimal format."""
    rows = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        header = f.readline()  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            # Format: datetime,000,integer_part,decimal_part,
            if len(parts) < 4:
                continue
            dt_str = parts[0]
            int_part = parts[2].strip()
            dec_part = parts[3].strip()
            if '(invalid)' in int_part or '(invalid)' in dec_part:
                continue
            try:
                temp = float(int_part + '.' + dec_part)
                dt = pd.to_datetime(dt_str, format='%d/%m/%Y %H:%M:%S')
                rows.append({'datetime': dt, 'temp_actual': temp})
            except (ValueError, TypeError):
                continue
    return pd.DataFrame(rows)


print("[*] Parsing actual temperature...")
temp_actual = parse_actual_temp(ACTUAL_TEMP_CSV)
print(f"[+] Actual temp: {len(temp_actual)} hourly rows")
print(f"    Period: {temp_actual['datetime'].min()} to {temp_actual['datetime'].max()}")

# Daily mean actual temperature
temp_actual['date'] = temp_actual['datetime'].dt.normalize()
daily_temp_actual = temp_actual.groupby('date')['temp_actual'].mean().reset_index()
daily_temp_actual.columns = ['date', 'temp_actual_daily']
print(f"[+] Daily actual temp: {len(daily_temp_actual)} days")
print(f"    Range: {daily_temp_actual['temp_actual_daily'].min():.1f} to {daily_temp_actual['temp_actual_daily'].max():.1f} C")


# --- 1c. GFS forecast temperature ---
def parse_gfs_temp(filepath):
    """Parse Tampretures.csv with both GFS and actual columns."""
    rows = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        header = f.readline()  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            # Format: dt1,000,gfs_int,gfs_dec,dt2,000,act_int,act_dec,
            if len(parts) < 8:
                continue
            dt_str = parts[0]
            gfs_int = parts[2].strip()
            gfs_dec = parts[3].strip()
            act_int = parts[6].strip()
            act_dec = parts[7].strip()

            try:
                dt = pd.to_datetime(dt_str, format='%d/%m/%Y %H:%M:%S')
            except ValueError:
                continue

            row = {'datetime': dt}

            # GFS forecast
            if '(invalid)' not in gfs_int and '(invalid)' not in gfs_dec:
                try:
                    row['temp_gfs'] = float(gfs_int + '.' + gfs_dec)
                except (ValueError, TypeError):
                    pass

            # Actual from this file
            if '(invalid)' not in act_int and '(invalid)' not in act_dec:
                try:
                    row['temp_actual_2'] = float(act_int + '.' + act_dec)
                except (ValueError, TypeError):
                    pass

            if 'temp_gfs' in row or 'temp_actual_2' in row:
                rows.append(row)
    return pd.DataFrame(rows)


print("[*] Parsing GFS forecast temperature...")
temp_gfs_raw = parse_gfs_temp(GFS_TEMP_CSV)
# Filter to only rows with GFS data
temp_gfs = temp_gfs_raw.dropna(subset=['temp_gfs']).copy()
print(f"[+] GFS forecast: {len(temp_gfs)} 15-min rows")
print(f"    Period: {temp_gfs['datetime'].min()} to {temp_gfs['datetime'].max()}")

# Aggregate GFS to daily mean
temp_gfs['date'] = temp_gfs['datetime'].dt.normalize()
daily_temp_gfs = temp_gfs.groupby('date')['temp_gfs'].mean().reset_index()
daily_temp_gfs.columns = ['date', 'temp_gfs_daily']
print(f"[+] Daily GFS temp: {len(daily_temp_gfs)} days")

# Also get daily actual from the GFS file for comparison
temp_gfs_act = temp_gfs_raw.dropna(subset=['temp_actual_2']).copy()
temp_gfs_act['date'] = temp_gfs_act['datetime'].dt.normalize()
daily_temp_gfs_act = temp_gfs_act.groupby('date')['temp_actual_2'].mean().reset_index()
daily_temp_gfs_act.columns = ['date', 'temp_actual_gfs_file']


# --- 1d. Merge everything ---
daily = daily_pnl.copy()
daily = daily.merge(daily_temp_actual, on='date', how='left')
daily = daily.merge(daily_temp_gfs, on='date', how='left')
daily = daily.merge(daily_temp_gfs_act, on='date', how='left')

# Year column
daily['year'] = daily['date'].dt.year

n_with_temp = daily['temp_actual_daily'].notna().sum()
n_with_gfs = daily['temp_gfs_daily'].notna().sum()
print(f"\n[+] Merged daily dataset: {len(daily)} trading days")
print(f"    With actual temp: {n_with_temp}")
print(f"    With GFS forecast: {n_with_gfs}")
print(f"    Date range: {daily['date'].min().date()} to {daily['date'].max().date()}")


# ============================================================================
# STEP 2: Test temperature filters
# ============================================================================
print("\n" + "=" * 70)
print("[*] STEP 2: Testing temperature filter variants")
print("=" * 70)


def compute_metrics(pnl_series, label=""):
    """Compute strategy metrics from a daily P&L series."""
    total_pnl = pnl_series.sum()
    n_days = len(pnl_series)
    mean_daily = pnl_series.mean() if n_days > 0 else 0
    std_daily = pnl_series.std() if n_days > 1 else 0
    sharpe = (mean_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0
    cum = pnl_series.cumsum()
    max_dd = (cum - cum.cummax()).min() if len(cum) > 0 else 0
    return {
        'label': label,
        'total_pnl': total_pnl,
        'n_days': n_days,
        'mean_daily': mean_daily,
        'sharpe': sharpe,
        'max_drawdown': max_dd
    }


def apply_filter(daily_df, cold_thresh, hot_thresh, temp_col='temp_actual_daily',
                 label="", use_lag=False):
    """
    Apply temperature filter. If use_lag=True, use yesterday's temp to decide today.
    Returns (trade_days_pnl, skip_days_pnl, metrics_dict).
    """
    df = daily_df.dropna(subset=[temp_col]).copy()
    df = df.sort_values('date').reset_index(drop=True)

    if use_lag:
        # Use yesterday's temp to decide today
        df['filter_temp'] = df[temp_col].shift(1)
        df = df.dropna(subset=['filter_temp'])
    else:
        df['filter_temp'] = df[temp_col]

    skip_mask = (df['filter_temp'] < cold_thresh) | (df['filter_temp'] > hot_thresh)
    trade = df[~skip_mask]['daily_pnl']
    skip = df[skip_mask]['daily_pnl']

    metrics = compute_metrics(trade, label)
    metrics['n_skipped'] = len(skip)
    metrics['mean_spread_trade'] = trade.mean() if len(trade) > 0 else 0
    metrics['mean_spread_skip'] = skip.mean() if len(skip) > 0 else 0

    return trade, skip, metrics


# Variant A: No filter (baseline)
trade_a = daily.dropna(subset=['temp_actual_daily']).sort_values('date')['daily_pnl']
metrics_a = compute_metrics(trade_a, "A. No filter (baseline)")
metrics_a['n_skipped'] = 0
metrics_a['mean_spread_trade'] = trade_a.mean()
metrics_a['mean_spread_skip'] = np.nan

# Variant B: Perfect foresight, skip < 0 or > 25
trade_b, skip_b, metrics_b = apply_filter(
    daily, cold_thresh=0, hot_thresh=25,
    label="B. Perfect: cold<0 OR hot>25")

# Variant C: Cold only < 0
trade_c, skip_c, metrics_c = apply_filter(
    daily, cold_thresh=0, hot_thresh=999,
    label="C. Perfect: cold<0 only")

# Variant D: Hot only > 25
trade_d, skip_d, metrics_d = apply_filter(
    daily, cold_thresh=-999, hot_thresh=25,
    label="D. Perfect: hot>25 only")

# Variant E: Tight: < 2 or > 23
trade_e, skip_e, metrics_e = apply_filter(
    daily, cold_thresh=2, hot_thresh=23,
    label="E. Perfect: cold<2 OR hot>23")

# Variant F: GFS forecast filter (Sep 2025 - Feb 2026 only)
gfs_subset = daily.dropna(subset=['temp_gfs_daily']).copy()
trade_f, skip_f, metrics_f = apply_filter(
    gfs_subset, cold_thresh=0, hot_thresh=25,
    temp_col='temp_gfs_daily',
    label="F. GFS forecast: cold<0 OR hot>25")

# Also compute baseline for the GFS period only for fair comparison
gfs_baseline = compute_metrics(
    gfs_subset.sort_values('date')['daily_pnl'],
    "F_baseline. No filter (GFS period)")
gfs_baseline['n_skipped'] = 0

all_metrics = [metrics_a, metrics_b, metrics_c, metrics_d, metrics_e, metrics_f]

print("\n--- Filter Comparison ---")
print(f"{'Variant':<40s} {'Total PnL':>10s} {'#Trade':>7s} {'#Skip':>6s} "
      f"{'Mean/day':>9s} {'Skip/day':>9s} {'Sharpe':>7s} {'MaxDD':>8s}")
print("-" * 100)
for m in all_metrics:
    skip_str = f"{m['mean_spread_skip']:>9.2f}" if not np.isnan(m.get('mean_spread_skip', float('nan'))) else "      N/A"
    print(f"{m['label']:<40s} {m['total_pnl']:>10.1f} {m['n_days']:>7d} {m['n_skipped']:>6d} "
          f"{m['mean_spread_trade']:>9.2f} {skip_str} "
          f"{m['sharpe']:>7.3f} {m['max_drawdown']:>8.1f}")

# Monthly breakdown for key variants
print("\n--- Monthly P&L Breakdown (EUR) ---")
for label_name, var_label, cold, hot in [
    ("No filter", "A", -999, 999),
    ("Cold<0 + Hot>25", "B", 0, 25),
    ("Cold only<0", "C", 0, 999),
    ("Tight<2 +>23", "E", 2, 23),
]:
    df_filt = daily.dropna(subset=['temp_actual_daily']).copy()
    df_filt = df_filt.sort_values('date')
    if cold > -999 or hot < 999:
        mask = (df_filt['temp_actual_daily'] < cold) | (df_filt['temp_actual_daily'] > hot)
        df_filt = df_filt[~mask]
    df_filt['month'] = df_filt['date'].dt.to_period('M')
    monthly = df_filt.groupby('month')['daily_pnl'].sum()
    print(f"\n  {var_label}. {label_name}:")
    for m_period, val in monthly.items():
        print(f"    {m_period}: {val:>10.1f} EUR")
    print(f"    TOTAL: {monthly.sum():>10.1f} EUR")


# ============================================================================
# STEP 3: Optimize thresholds (2024 in-sample, 2025+ out-of-sample)
# ============================================================================
print("\n" + "=" * 70)
print("[*] STEP 3: Threshold optimization")
print("=" * 70)

cold_range = np.arange(-3, 5.5, 0.5)
hot_range = np.arange(20, 31, 1.0)

# Split data
daily_valid = daily.dropna(subset=['temp_actual_daily']).sort_values('date').copy()
insample = daily_valid[daily_valid['year'] == 2024].copy()
outsample = daily_valid[daily_valid['year'] >= 2025].copy()

print(f"[+] In-sample (2024): {len(insample)} days")
print(f"[+] Out-of-sample (2025+): {len(outsample)} days")

# Heatmaps
pnl_heatmap_is = np.zeros((len(cold_range), len(hot_range)))
sharpe_heatmap_is = np.zeros((len(cold_range), len(hot_range)))
pnl_heatmap_oos = np.zeros((len(cold_range), len(hot_range)))
sharpe_heatmap_oos = np.zeros((len(cold_range), len(hot_range)))

for i, cold_t in enumerate(cold_range):
    for j, hot_t in enumerate(hot_range):
        # In-sample
        mask_is = (insample['temp_actual_daily'] < cold_t) | (insample['temp_actual_daily'] > hot_t)
        trade_is = insample[~mask_is]['daily_pnl']
        m_is = compute_metrics(trade_is)
        pnl_heatmap_is[i, j] = m_is['total_pnl']
        sharpe_heatmap_is[i, j] = m_is['sharpe']

        # Out-of-sample
        mask_oos = (outsample['temp_actual_daily'] < cold_t) | (outsample['temp_actual_daily'] > hot_t)
        trade_oos = outsample[~mask_oos]['daily_pnl']
        m_oos = compute_metrics(trade_oos)
        pnl_heatmap_oos[i, j] = m_oos['total_pnl']
        sharpe_heatmap_oos[i, j] = m_oos['sharpe']

# Best in-sample
best_idx_is = np.unravel_index(np.argmax(pnl_heatmap_is), pnl_heatmap_is.shape)
best_cold_is = cold_range[best_idx_is[0]]
best_hot_is = hot_range[best_idx_is[1]]
best_pnl_is = pnl_heatmap_is[best_idx_is]
best_sharpe_is = sharpe_heatmap_is[best_idx_is]

# OOS performance of the IS-optimal thresholds
oos_pnl_at_is_best = pnl_heatmap_oos[best_idx_is]
oos_sharpe_at_is_best = sharpe_heatmap_oos[best_idx_is]

# Best OOS (hindsight)
best_idx_oos = np.unravel_index(np.argmax(pnl_heatmap_oos), pnl_heatmap_oos.shape)
best_cold_oos = cold_range[best_idx_oos[0]]
best_hot_oos = hot_range[best_idx_oos[1]]

# Baseline (no filter) for each period
baseline_is_pnl = insample['daily_pnl'].sum()
baseline_oos_pnl = outsample['daily_pnl'].sum()
baseline_is_sharpe = compute_metrics(insample['daily_pnl'])['sharpe']
baseline_oos_sharpe = compute_metrics(outsample['daily_pnl'])['sharpe']

print(f"\n--- In-Sample Optimal (2024) ---")
print(f"  Best cold threshold: {best_cold_is:.1f} C")
print(f"  Best hot threshold:  {best_hot_is:.1f} C")
print(f"  IS P&L:    {best_pnl_is:>10.1f} EUR  (baseline: {baseline_is_pnl:.1f})")
print(f"  IS Sharpe: {best_sharpe_is:>10.3f}      (baseline: {baseline_is_sharpe:.3f})")
print(f"\n--- Out-of-Sample (2025+) with IS-optimal thresholds ---")
print(f"  OOS P&L:    {oos_pnl_at_is_best:>10.1f} EUR  (baseline: {baseline_oos_pnl:.1f})")
print(f"  OOS Sharpe: {oos_sharpe_at_is_best:>10.3f}      (baseline: {baseline_oos_sharpe:.3f})")
print(f"\n--- OOS Best (hindsight) ---")
print(f"  Best cold: {best_cold_oos:.1f} C, Best hot: {best_hot_oos:.1f} C")
print(f"  OOS P&L:   {pnl_heatmap_oos[best_idx_oos]:>10.1f} EUR")
print(f"  OOS Sharpe: {sharpe_heatmap_oos[best_idx_oos]:>10.3f}")


# ============================================================================
# STEP 4: GFS forecast quality check
# ============================================================================
print("\n" + "=" * 70)
print("[*] STEP 4: GFS forecast quality check")
print("=" * 70)

# Merge GFS daily with actual daily for the overlapping period
gfs_check = daily_temp_gfs.merge(daily_temp_actual, on='date', how='inner')
print(f"[+] Overlapping days (GFS vs actual): {len(gfs_check)}")
if len(gfs_check) > 0:
    corr = gfs_check['temp_gfs_daily'].corr(gfs_check['temp_actual_daily'])
    mae = (gfs_check['temp_gfs_daily'] - gfs_check['temp_actual_daily']).abs().mean()
    bias = (gfs_check['temp_gfs_daily'] - gfs_check['temp_actual_daily']).mean()
    print(f"  Correlation: {corr:.4f}")
    print(f"  MAE:         {mae:.2f} C")
    print(f"  Bias:        {bias:.2f} C")

    # Filter agreement
    cold_t, hot_t = 0, 25
    actual_skip = (gfs_check['temp_actual_daily'] < cold_t) | (gfs_check['temp_actual_daily'] > hot_t)
    gfs_skip = (gfs_check['temp_gfs_daily'] < cold_t) | (gfs_check['temp_gfs_daily'] > hot_t)
    agree = (actual_skip == gfs_skip).sum()
    disagree = (actual_skip != gfs_skip).sum()
    print(f"\n  Filter agreement (cold<0, hot>25):")
    print(f"    Agree:    {agree} days ({agree/len(gfs_check)*100:.1f}%)")
    print(f"    Disagree: {disagree} days ({disagree/len(gfs_check)*100:.1f}%)")
    print(f"    Actual skips: {actual_skip.sum()}, GFS skips: {gfs_skip.sum()}")

    # Check both-skip, false-skip, missed-skip
    both_skip = (actual_skip & gfs_skip).sum()
    false_skip = (~actual_skip & gfs_skip).sum()
    missed_skip = (actual_skip & ~gfs_skip).sum()
    print(f"    Both skip:   {both_skip} (correctly avoided)")
    print(f"    False skip:  {false_skip} (GFS skips but actual is fine)")
    print(f"    Missed skip: {missed_skip} (GFS trades but actual is extreme)")


# ============================================================================
# STEP 5: Visualization
# ============================================================================
print("\n" + "=" * 70)
print("[*] STEP 5: Creating visualization")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Temperature-Filtered "Sell DA, Buy Imbalance" Backtest', fontsize=14, fontweight='bold')

# --- Panel 1: Cumulative P&L curves ---
ax1 = axes[0, 0]

def get_cum_pnl_series(daily_df, cold_thresh, hot_thresh, temp_col='temp_actual_daily'):
    """Get cumulative P&L as a time series."""
    df = daily_df.dropna(subset=[temp_col]).sort_values('date').copy()
    if cold_thresh > -900 or hot_thresh < 900:
        mask = (df[temp_col] < cold_thresh) | (df[temp_col] > hot_thresh)
        df.loc[mask, 'daily_pnl'] = 0  # zero out skipped days for cumulative view
    return df['date'], df['daily_pnl'].cumsum()

d_a, c_a = get_cum_pnl_series(daily, -999, 999)
d_b, c_b = get_cum_pnl_series(daily, 0, 25)
d_c, c_c = get_cum_pnl_series(daily, 0, 999)
d_d, c_d = get_cum_pnl_series(daily, -999, 25)

ax1.plot(d_a, c_a, 'k-', linewidth=1.5, label='A. No filter', alpha=0.8)
ax1.plot(d_b, c_b, 'b-', linewidth=1.5, label='B. Cold<0 + Hot>25', alpha=0.8)
ax1.plot(d_c, c_c, 'r-', linewidth=1.5, label='C. Cold<0 only', alpha=0.8)
ax1.plot(d_d, c_d, 'g-', linewidth=1.5, label='D. Hot>25 only', alpha=0.8)
ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('Date')
ax1.set_ylabel('Cumulative P&L (EUR, 1MW)')
ax1.set_title('Cumulative P&L: Perfect Foresight Filter Variants')
ax1.legend(fontsize=8, loc='best')
ax1.grid(True, alpha=0.3)

# --- Panel 2: In-sample heatmap (2024) ---
ax2 = axes[0, 1]
im2 = ax2.imshow(pnl_heatmap_is, aspect='auto', origin='lower',
                  cmap='RdYlGn',
                  extent=[hot_range[0]-0.5, hot_range[-1]+0.5,
                          cold_range[0]-0.25, cold_range[-1]+0.25])
ax2.set_xlabel('Hot threshold (C)')
ax2.set_ylabel('Cold threshold (C)')
ax2.set_title(f'2024 In-Sample P&L (EUR)\nBest: cold<{best_cold_is:.1f}, hot>{best_hot_is:.0f}')
plt.colorbar(im2, ax=ax2, label='Total P&L (EUR)')
# Mark the best point
ax2.plot(best_hot_is, best_cold_is, 'k*', markersize=15, markeredgecolor='white',
         markeredgewidth=1)

# --- Panel 3: Out-of-sample heatmap (2025+) ---
ax3 = axes[1, 0]
im3 = ax3.imshow(pnl_heatmap_oos, aspect='auto', origin='lower',
                  cmap='RdYlGn',
                  extent=[hot_range[0]-0.5, hot_range[-1]+0.5,
                          cold_range[0]-0.25, cold_range[-1]+0.25])
ax3.set_xlabel('Hot threshold (C)')
ax3.set_ylabel('Cold threshold (C)')
ax3.set_title(f'2025+ Out-of-Sample P&L (EUR)\nIS-optimal applied: cold<{best_cold_is:.1f}, hot>{best_hot_is:.0f}')
plt.colorbar(im3, ax=ax3, label='Total P&L (EUR)')
# Mark IS-optimal point on OOS
ax3.plot(best_hot_is, best_cold_is, 'k*', markersize=15, markeredgecolor='white',
         markeredgewidth=1)
# Mark OOS-best
ax3.plot(best_hot_oos, best_cold_oos, 'r*', markersize=12, markeredgecolor='white',
         markeredgewidth=1)

# --- Panel 4: GFS vs Actual scatter ---
ax4 = axes[1, 1]
if len(gfs_check) > 0:
    ax4.scatter(gfs_check['temp_actual_daily'], gfs_check['temp_gfs_daily'],
                alpha=0.6, s=20, c='steelblue', edgecolors='none')
    # Perfect forecast line
    t_min = min(gfs_check['temp_actual_daily'].min(), gfs_check['temp_gfs_daily'].min()) - 2
    t_max = max(gfs_check['temp_actual_daily'].max(), gfs_check['temp_gfs_daily'].max()) + 2
    ax4.plot([t_min, t_max], [t_min, t_max], 'k--', alpha=0.5, label='Perfect forecast')
    # Threshold lines
    ax4.axvline(0, color='blue', linestyle=':', alpha=0.7, label='Cold thresh (0 C)')
    ax4.axhline(0, color='blue', linestyle=':', alpha=0.7)
    ax4.axvline(25, color='red', linestyle=':', alpha=0.7, label='Hot thresh (25 C)')
    ax4.axhline(25, color='red', linestyle=':', alpha=0.7)
    ax4.set_xlabel('Actual daily mean temp (C)')
    ax4.set_ylabel('GFS forecast daily mean temp (C)')
    ax4.set_title(f'GFS vs Actual (Sep 2025-Feb 2026)\nr={corr:.3f}, MAE={mae:.2f} C')
    ax4.legend(fontsize=8, loc='upper left')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(t_min, t_max)
    ax4.set_ylim(t_min, t_max)
else:
    ax4.text(0.5, 0.5, 'No overlapping GFS/Actual data', ha='center', va='center',
             transform=ax4.transAxes)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
print(f"[+] Saved figure to: {OUT_PNG}")


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("[*] FINAL SUMMARY")
print("=" * 70)

print(f"""
Strategy: Sell 1 MWh/h at DA price, buy at imbalance settlement price.
Period:   {daily['date'].min().date()} to {daily['date'].max().date()} ({len(daily)} trading days)

--- Perfect Foresight Results (full period) ---
  A. No filter:         {metrics_a['total_pnl']:>10.1f} EUR, Sharpe={metrics_a['sharpe']:.3f}
  B. Skip cold<0+hot>25:{metrics_b['total_pnl']:>10.1f} EUR, Sharpe={metrics_b['sharpe']:.3f}, skip {metrics_b['n_skipped']} days
  C. Skip cold<0 only:  {metrics_c['total_pnl']:>10.1f} EUR, Sharpe={metrics_c['sharpe']:.3f}, skip {metrics_c['n_skipped']} days
  D. Skip hot>25 only:  {metrics_d['total_pnl']:>10.1f} EUR, Sharpe={metrics_d['sharpe']:.3f}, skip {metrics_d['n_skipped']} days
  E. Skip cold<2+hot>23:{metrics_e['total_pnl']:>10.1f} EUR, Sharpe={metrics_e['sharpe']:.3f}, skip {metrics_e['n_skipped']} days

--- Threshold Optimization ---
  IS-optimal (2024):     cold < {best_cold_is:.1f} C, hot > {best_hot_is:.0f} C
  IS P&L:  {best_pnl_is:.1f} EUR (baseline {baseline_is_pnl:.1f}), Sharpe={best_sharpe_is:.3f} (baseline {baseline_is_sharpe:.3f})
  OOS P&L: {oos_pnl_at_is_best:.1f} EUR (baseline {baseline_oos_pnl:.1f}), Sharpe={oos_sharpe_at_is_best:.3f} (baseline {baseline_oos_sharpe:.3f})

--- GFS Forecast Filter (Sep 2025 - Feb 2026) ---
  GFS-filtered P&L: {metrics_f['total_pnl']:>10.1f} EUR, Sharpe={metrics_f['sharpe']:.3f}
  GFS period baseline: {gfs_baseline['total_pnl']:.1f} EUR

--- Key Question: Does skipping extreme temp days help? ---
  Mean daily P&L on skipped days (B): {metrics_b['mean_spread_skip']:.2f} EUR
  Mean daily P&L on traded days (B):  {metrics_b['mean_spread_trade']:.2f} EUR
""")

print("[+] Done.")
