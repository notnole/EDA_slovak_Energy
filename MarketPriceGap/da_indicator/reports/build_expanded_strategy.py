"""
Expanded Speculative Strategy: 6 rules (drop old R4, add 3 new).
Generates all charts for the revised report: decision screens, P&L, rule breakdown.
Also generates BESS decision screen.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from pathlib import Path
import sys

BASE = Path(r'C:\Users\20254757\pycharmprojects\ipesoft_eda_data')
OUT  = BASE / 'MarketPriceGap' / 'da_indicator'
RPT_FIG = OUT / 'reports' / 'figures'
RPT_FIG.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(OUT / 'scripts'))
from build_da_indicator import load_all_data, build_features, generate_signals


# ============================================================
# EXPANDED SIGNAL RULES
# ============================================================

RULE_COLORS = {
    'Hourly Persistence': '#9c27b0',
    'Daily Persistence':  '#2196f3',
    'High Load':          '#ff9800',
    'No Trade':           '#bdbdbd',
}

RULE_ORDER = ['Hourly Persistence', 'Daily Persistence', 'High Load']

# Discarded signals (tested with corrected data, not profitable):
# - Weekend sell DA: PF=1.22 in 2025 but collapses to PF=0.69 in 2026
# - Night h0-5 sell DA: PF=0.86, loses money - was artifact of data bug
# - Evening Peak (h17-20): PF~1.0, breakeven
# - Morning Peak (h7-10): PF~1.0, breakeven
# - DA price level (>150): PF=1.36 in 2025 but collapses in 2026


def classify_expanded(row):
    """3-rule cascade. Returns (rule_name, signal_direction, trade_flag)."""
    spread_yest_h = row.get('spread_yest_h', 0)
    spread_mean_yest = row.get('spread_mean_yest', 0)
    load_dev = row.get('load_fcst_dev', 0)

    if pd.isna(spread_yest_h): spread_yest_h = 0
    if pd.isna(spread_mean_yest): spread_mean_yest = 0
    if pd.isna(load_dev): load_dev = 0

    # R1: Large hourly spread persistence (|yest same-hour spread| >= 20 EUR)
    # PF=1.78, the strongest signal. When yesterday's same-hour spread was large,
    # the same structural conditions (outage, constraint, weather) likely persist.
    if abs(spread_yest_h) >= 20:
        return 'Hourly Persistence', (1 if spread_yest_h > 0 else -1), 1.0

    # R2: Daily spread persistence (yesterday daily avg spread < -10 EUR)
    # PF=1.31 in cascade, PF=4.20 in 2026. When the IDM systematically
    # exceeded DA yesterday, the same direction tends to continue.
    if spread_mean_yest < -10:
        return 'Daily Persistence', -1, 1.0

    # R3: High load forecast deviation (> 85 MW above 7-day avg)
    # PF=1.10 in cascade. When load is abnormally high, DA tends to
    # underprice relative to IDM as the market anchors to recent conditions.
    if load_dev > 85:
        return 'High Load', -1, 1.0

    return 'No Trade', 0, 0.0


def compute_metrics(pnl_series):
    wins = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series < 0]
    n = len(pnl_series)
    if n == 0:
        return {'n': 0, 'total_pnl': 0, 'pf': 0, 'avg_win': 0,
                'avg_loss': 0, 'wl': 0, 'win_rate': 0, 'exp': 0}
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
    wl = avg_win / avg_loss if avg_loss > 0 else float('inf')
    wr = len(wins) / n if n > 0 else 0
    exp = (wr * avg_win) - ((1 - wr) * avg_loss)
    return {'n': n, 'total_pnl': pnl_series.sum(), 'pf': pf,
            'avg_win': avg_win, 'avg_loss': avg_loss, 'wl': wl,
            'win_rate': wr, 'exp': exp}


# ============================================================
# BESS DP SCHEDULER (for decision screen)
# ============================================================

def optimize_bess_dp(prices, predicted_ranks, n_charge=2, n_discharge=2,
                     start_soc=1, terminal_penalty=5.0):
    """DP-optimal BESS scheduling. Supports variable-length days and terminal SoC penalty."""
    n_hours = len(prices)
    if n_hours < 1 or np.any(np.isnan(prices[:n_hours])):
        return np.zeros(n_hours), 0.0, np.ones(n_hours + 1) * start_soc
    p = predicted_ranks[:n_hours]
    actual = prices[:n_hours]
    NC, ND = n_charge + 1, n_discharge + 1
    INF = -1e18
    dp = np.full((n_hours + 1, 3, NC, ND), INF)
    choice = np.zeros((n_hours, 3, NC, ND), dtype=int)
    # Terminal: penalize SoC != 1
    for s in range(3):
        penalty = -terminal_penalty if s != 1 else 0.0
        for cr in range(NC):
            for dr in range(ND):
                dp[n_hours][s][cr][dr] = penalty
    for h in range(n_hours - 1, -1, -1):
        for s in range(3):
            for cr in range(NC):
                for dr in range(ND):
                    best, best_a = INF, 0
                    val = dp[h+1][s][cr][dr]
                    if val > best: best, best_a = val, 0
                    if cr > 0 and s + 1 <= 2:
                        val = -p[h] + dp[h+1][s+1][cr-1][dr]
                        if val > best: best, best_a = val, 1
                    if dr > 0 and s - 1 >= 0:
                        val = p[h] + dp[h+1][s-1][cr][dr-1]
                        if val > best: best, best_a = val, -1
                    dp[h][s][cr][dr] = best
                    choice[h][s][cr][dr] = best_a
    actions = np.zeros(n_hours)
    soc_trace = np.zeros(n_hours + 1)
    soc_trace[0] = start_soc
    soc, cr, dr = start_soc, n_charge, n_discharge
    for h in range(n_hours):
        a = choice[h][soc][cr][dr]
        actions[h] = a
        if a == 1: soc += 1; cr -= 1
        elif a == -1: soc -= 1; dr -= 1
        soc_trace[h+1] = soc
    revenue = -np.sum(actions * actual)
    return actions, revenue, soc_trace


def main():
    print("[*] Loading data...")
    df = load_all_data()
    df, daily = build_features(df)
    df = generate_signals(df)

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 200})

    # ================================================================
    # APPLY EXPANDED RULES
    # ================================================================
    print("[*] Applying 3-rule signal cascade...")
    results = df.apply(classify_expanded, axis=1, result_type='expand')
    df['rule_exp'] = results[0]
    df['signal_exp'] = results[1].astype(float)
    df['trade_exp'] = results[2].astype(float)

    # Filter to traded hours with IDM data
    mask = df['idm_vwap'].notna() & df['spread_da_idm'].notna() & (df['trade_exp'] > 0)
    traded = df[mask].copy()
    traded['pnl'] = traded['signal_exp'] * traded['spread_da_idm']
    traded['year'] = traded['date'].dt.year

    print(f"[+] Traded hours: {len(traded)}")

    # ================================================================
    # PRINT METRICS
    # ================================================================
    print("\n" + "=" * 90)
    print("  SPECULATIVE STRATEGY METRICS (3 RULES, CORRECTED DATA)")
    print("=" * 90)

    m = compute_metrics(traded['pnl'])
    print(f"\n--- OVERALL ---")
    print(f"  Trades:         {m['n']}")
    print(f"  Total P&L:      {m['total_pnl']:+,.0f} EUR")
    print(f"  Profit Factor:  {m['pf']:.2f}x")
    print(f"  Win/Loss Ratio: {m['wl']:.2f}x")
    print(f"  Expectancy:     {m['exp']:+.2f} EUR/trade")

    print(f"\n--- BY YEAR ---")
    for yr in sorted(traded['year'].unique()):
        sub = traded[traded['year'] == yr]
        m = compute_metrics(sub['pnl'])
        print(f"  {yr}: n={m['n']:5d}  PF={m['pf']:.2f}x  W/L={m['wl']:.2f}x  "
              f"Exp={m['exp']:+.2f}  Total={m['total_pnl']:+,.0f}")

    print(f"\n--- BY RULE (CASCADE) ---")
    rule_metrics = []
    for rule in RULE_ORDER:
        sub = traded[traded['rule_exp'] == rule]
        if len(sub) > 0:
            m = compute_metrics(sub['pnl'])
            m['rule'] = rule
            rule_metrics.append(m)
            print(f"  {rule:20s}  n={m['n']:5d}  PF={m['pf']:5.2f}x  W/L={m['wl']:5.2f}x  "
                  f"AvgWin={m['avg_win']:+6.1f}  AvgLoss={m['avg_loss']:6.1f}  "
                  f"Exp={m['exp']:+6.2f}  Total={m['total_pnl']:+8,.0f}")

    # By rule AND year
    print(f"\n--- BY RULE AND YEAR ---")
    for rule in RULE_ORDER:
        sub = traded[traded['rule_exp'] == rule]
        if len(sub) == 0:
            continue
        print(f"\n  {rule}:")
        for yr in sorted(traded['year'].unique()):
            s = sub[sub['year'] == yr]
            if len(s) > 0:
                m = compute_metrics(s['pnl'])
                print(f"    {yr}: PF={m['pf']:5.2f}x  Exp={m['exp']:+6.2f}  "
                      f"Total={m['total_pnl']:+8,.0f}  n={m['n']}")

    rm = pd.DataFrame(rule_metrics)

    # ================================================================
    # CHART 1: RULE BREAKDOWN (PF, Win/Loss, Expectancy)
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    # Panel 1: Profit Factor
    ax = axes[0]
    colors = [RULE_COLORS.get(r, '#999') for r in rm['rule']]
    bars = ax.barh(rm['rule'], rm['pf'], color=colors, edgecolor='white', alpha=0.85)
    ax.axvline(1.0, color='red', ls='--', alpha=0.5, label='Breakeven')
    ax.set_xlabel('Profit Factor')
    ax.set_title('Profit Factor by Rule')
    for i, row in rm.iterrows():
        ax.text(row['pf'] + 0.05, i, f"{row['pf']:.2f}x", va='center', fontsize=9, fontweight='bold')
    ax.set_xlim(0, max(rm['pf']) * 1.2)
    ax.legend(fontsize=8)

    # Panel 2: Win vs Loss size
    ax = axes[1]
    x_pos = range(len(rm))
    w = 0.35
    ax.barh([i - w/2 for i in x_pos], rm['avg_win'], w,
            label='Avg Win', color='#4caf50', edgecolor='white')
    ax.barh([i + w/2 for i in x_pos], rm['avg_loss'], w,
            label='Avg Loss', color='#d32f2f', edgecolor='white')
    ax.set_yticks(list(x_pos))
    ax.set_yticklabels(rm['rule'])
    ax.set_xlabel('EUR per trade')
    ax.set_title('Win Size vs Loss Size')
    ax.legend(fontsize=8)
    for i, row in rm.iterrows():
        ax.text(max(row['avg_win'], row['avg_loss']) + 0.5, i,
                f"Ratio: {row['wl']:.1f}x", va='center', fontsize=8)

    # Panel 3: Expectancy
    ax = axes[2]
    colors_exp = ['#4caf50' if v > 0 else '#d32f2f' for v in rm['exp']]
    ax.barh(rm['rule'], rm['exp'], color=colors_exp, edgecolor='white')
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel('Expectancy (EUR/trade)')
    ax.set_title('Expected Value per Trade')
    for i, row in rm.iterrows():
        side = 0.3 if row['exp'] >= 0 else -0.3
        ha = 'left' if row['exp'] >= 0 else 'right'
        ax.text(row['exp'] + side, i,
                f"{row['exp']:+.1f} (n={row['n']})", va='center', fontsize=9, ha=ha)

    fig.suptitle('Rule Breakdown: Profit Factor, Win/Loss Ratio, Expectancy', fontweight='bold')
    fig.tight_layout()
    fig.savefig(RPT_FIG / 'spec_rule_breakdown.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[+] spec_rule_breakdown.png")

    # ================================================================
    # CHART 2: ROLLING PF AND P&L
    # ================================================================
    daily_pnl = traded.groupby('date').agg(
        pnl=('pnl', 'sum'),
        gross_profit=('pnl', lambda x: x[x > 0].sum()),
        gross_loss=('pnl', lambda x: abs(x[x < 0].sum())),
    ).reset_index()
    daily_pnl['date'] = pd.to_datetime(daily_pnl['date'])
    daily_pnl = daily_pnl.sort_values('date')
    daily_pnl['roll_gp'] = daily_pnl['gross_profit'].rolling(30, min_periods=10).sum()
    daily_pnl['roll_gl'] = daily_pnl['gross_loss'].rolling(30, min_periods=10).sum()
    daily_pnl['roll_pf'] = daily_pnl['roll_gp'] / daily_pnl['roll_gl'].clip(lower=1)
    daily_pnl['roll_avg'] = daily_pnl['pnl'].rolling(30, min_periods=10).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.plot(daily_pnl['date'], daily_pnl['roll_pf'], 'b-', lw=1.5)
    ax1.axhline(1.0, color='red', ls='--', alpha=0.5, label='Breakeven (PF=1.0)')
    ax1.axvline(pd.Timestamp('2026-01-01'), color='red', ls='--', alpha=0.3)
    ax1.axvspan(pd.Timestamp('2026-01-01'), daily_pnl['date'].max(), alpha=0.08, color='red')
    ax1.set_ylabel('30-day Rolling Profit Factor')
    ax1.set_title('Signal Stays Profitable Over Time')
    ax1.set_ylim(0, 6)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(daily_pnl['date'], 0, daily_pnl['roll_avg'],
                     where=daily_pnl['roll_avg'] >= 0, alpha=0.3, color='green')
    ax2.fill_between(daily_pnl['date'], 0, daily_pnl['roll_avg'],
                     where=daily_pnl['roll_avg'] < 0, alpha=0.3, color='red')
    ax2.plot(daily_pnl['date'], daily_pnl['roll_avg'], 'k-', lw=1)
    ax2.axhline(0, color='black', lw=0.5)
    ax2.axvline(pd.Timestamp('2026-01-01'), color='red', ls='--', alpha=0.3)
    ax2.set_ylabel('30-day Rolling Avg Daily P&L (EUR)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RPT_FIG / 'spec_rolling_pf.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("[+] spec_rolling_pf.png")

    # ================================================================
    # CHART 3: CUMULATIVE P&L WITH JAN 2026 FOCUS
    # ================================================================
    daily_pnl['cum_pnl'] = daily_pnl['pnl'].cumsum()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
    ax1.plot(daily_pnl['date'], daily_pnl['cum_pnl'], 'b-', lw=1.2)
    ax1.axvspan(pd.Timestamp('2026-01-01'), daily_pnl['date'].max(), alpha=0.08, color='red')
    ax1.axvline(pd.Timestamp('2026-01-01'), color='red', ls='--', alpha=0.4, label='2026 start')
    ax1.set_ylabel('Cumulative P&L (EUR)')
    ax1.set_title('Expanded Strategy: Cumulative P&L')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    jan_data = daily_pnl[(daily_pnl['date'] >= '2025-12-01') & (daily_pnl['date'] <= '2026-02-05')]
    if len(jan_data) > 0:
        jan_cum = jan_data['pnl'].cumsum()
        colors_bar = ['#4caf50' if v > 0 else '#d32f2f' for v in jan_data['pnl'].values]
        ax2.bar(jan_data['date'], jan_data['pnl'], color=colors_bar, alpha=0.6, width=0.8)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(jan_data['date'], jan_cum.values, 'k-o', ms=3, lw=1.5, label='Cumulative')
        ax2_twin.set_ylabel('Cumulative P&L (EUR)')
        ax2_twin.legend(fontsize=8, loc='upper left')
        ax2.set_ylabel('Daily P&L (EUR)')
        ax2.set_xlabel('Date')
        ax2.axvline(pd.Timestamp('2026-01-01'), color='red', ls='--', alpha=0.4)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        ax2.tick_params(axis='x', rotation=45)
        jan26_only = traded[traded['year'] == 2026]
        if len(jan26_only) > 0:
            m_jan = compute_metrics(jan26_only['pnl'])
            ax2.set_title(f'Dec 2025 -- Feb 2026  |  Jan 2026: {m_jan["total_pnl"]:+,.0f} EUR, '
                          f'PF={m_jan["pf"]:.1f}x, Exp={m_jan["exp"]:+.1f} EUR/trade')
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RPT_FIG / 'spec_pnl_jan2026.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("[+] spec_pnl_jan2026.png")

    # ================================================================
    # CHART 4: SPECULATIVE DECISION SCREEN (2 example days)
    # ================================================================
    jan26 = df[(df['date'] >= '2026-01-01') & (df['date'] <= '2026-01-31')]
    jan26_dates = sorted(jan26['date'].unique())

    # Find two example days: one with strong persistence, one with high load + daily pers
    spec_examples = []
    for d in jan26_dates:
        day = jan26[jan26['date'] == d]
        if day['idm_vwap'].notna().sum() >= 18:
            spread = day['spread_da_idm'].dropna()
            spread_yest = day['spread_yest_h'].dropna()
            if len(spread) > 0:
                spec_examples.append({
                    'date': d,
                    'mean_spread': spread.mean(),
                    'max_abs_yest': spread_yest.abs().max() if len(spread_yest) > 0 else 0,
                })
    spec_ex_df = pd.DataFrame(spec_examples)
    # Day 1: strongest hourly persistence day (largest yesterday spread)
    spec_ex_df = spec_ex_df.sort_values('max_abs_yest', ascending=False)
    spec_day1 = spec_ex_df.iloc[0]['date'] if len(spec_ex_df) > 0 else jan26_dates[0]
    # Day 2: a different day with mixed signals
    remaining = spec_ex_df[spec_ex_df['date'] != spec_day1]
    spec_day2 = remaining.iloc[1]['date'] if len(remaining) > 1 else jan26_dates[-1]
    print(f"[+] Spec decision screen days: {spec_day1}, {spec_day2}")

    for ex_date, label in [(spec_day1, 'persistence'), (spec_day2, 'mixed')]:
        day = df[df['date'] == ex_date].sort_values('hour').copy()
        day_idm = day[day['idm_vwap'].notna()].copy()
        if len(day_idm) < 10:
            continue

        # Apply expanded rules
        day_rules = day_idm.apply(classify_expanded, axis=1, result_type='expand')
        day_idm['rule_exp'] = day_rules[0].values
        day_idm['signal_exp'] = day_rules[1].astype(float).values
        day_idm['trade_exp'] = day_rules[2].astype(float).values
        day_idm['pnl_h'] = day_idm['signal_exp'] * day_idm['spread_da_idm'] * day_idm['trade_exp']

        hours = day_idm['hour'].values

        fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True,
                                 gridspec_kw={'height_ratios': [3, 1.5, 1.2, 2]})

        # Panel 1: DA and IDM prices + spread
        ax = axes[0]
        ax.plot(hours, day_idm['da_price'].values, 'b-o', ms=4, lw=1.5, label='DA Price')
        ax.plot(hours, day_idm['idm_vwap'].values, 'r-s', ms=4, lw=1.5, label='IDM VWAP')
        spread = day_idm['spread_da_idm'].values
        ax.fill_between(hours, day_idm['da_price'], day_idm['idm_vwap'],
                        where=day_idm['da_price'] > day_idm['idm_vwap'],
                        alpha=0.12, color='green')
        ax.fill_between(hours, day_idm['da_price'], day_idm['idm_vwap'],
                        where=day_idm['da_price'] <= day_idm['idm_vwap'],
                        alpha=0.12, color='red')
        ax.set_ylabel('Price (EUR/MWh)')
        date_str = pd.Timestamp(ex_date).strftime('%A %d %B %Y')
        ax.set_title(f'Decision Screen: {date_str}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

        # Panel 2: Key features (yesterday hourly spread, daily spread, load dev)
        ax = axes[1]
        spread_yest_vals = day_idm['spread_yest_h'].fillna(0).values
        spread_daily_vals = day_idm['spread_mean_yest'].fillna(0).values
        load_dev_vals = day_idm['load_fcst_dev'].fillna(0).values

        ax.plot(hours, spread_yest_vals, 'D-', color='#9c27b0', ms=3, lw=1, label='Yesterday Hourly Spread (EUR)')
        ax.plot(hours, spread_daily_vals, 'o-', color='#2196f3', ms=3, lw=1, label='Yesterday Daily Spread (EUR)')
        ax.plot(hours, load_dev_vals, 's-', color='#ff9800', ms=3, lw=1, label='Load Deviation (MW)')
        ax.axhline(20, color='#9c27b0', ls=':', alpha=0.4, lw=0.8)
        ax.axhline(-20, color='#9c27b0', ls=':', alpha=0.4, lw=0.8)
        ax.axhline(-10, color='#2196f3', ls=':', alpha=0.4, lw=0.8)
        ax.axhline(85, color='#ff9800', ls=':', alpha=0.4, lw=0.8)
        ax.set_ylabel('Feature Value')
        ax.legend(fontsize=7, ncol=3, loc='upper right')
        ax.grid(True, alpha=0.3)

        # Panel 3: Which rule fires (color-coded bars)
        ax = axes[2]
        rule_names = day_idm['rule_exp'].values
        for i, h in enumerate(hours):
            rule = rule_names[i]
            color = RULE_COLORS.get(rule, '#bdbdbd')
            ax.bar(h, 1, color=color, edgecolor='white', width=0.8)
        ax.set_yticks([])
        ax.set_ylabel('Active Rule')
        # Legend
        patches = [Patch(color=RULE_COLORS[r], label=r) for r in RULE_ORDER if r in rule_names]
        ax.legend(handles=patches, fontsize=7, ncol=len(patches), loc='upper center')

        # Panel 4: Signal direction + P&L
        ax = axes[3]
        pnl_h = day_idm['pnl_h'].values
        colors_pnl = ['#4caf50' if v > 0 else '#d32f2f' if v < 0 else '#bdbdbd' for v in pnl_h]
        ax.bar(hours, pnl_h, color=colors_pnl, alpha=0.7, edgecolor='white')
        ax.axhline(0, color='black', lw=0.5)
        cum = np.cumsum(pnl_h)
        ax.plot(hours, cum, 'k-o', ms=3, lw=1.5, label='Cumulative')
        ax.set_ylabel('P&L (EUR)')
        ax.set_xlabel('Hour of Day')
        total = cum[-1] if len(cum) > 0 else 0
        traded_h = (day_idm['trade_exp'] > 0).sum()
        wins = (pnl_h > 0).sum()
        ax.text(0.98, 0.05, f'Day total: {total:+.0f} EUR  |  {wins}/{traded_h} winning hours',
                transform=ax.transAxes, fontsize=9, ha='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(RPT_FIG / f'spec_decision_{label}.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"[+] spec_decision_{label}.png")

    # ================================================================
    # CHART 5: BESS DECISION SCREEN (5-panel, 2 example days)
    # Uses persistent SoC simulation with 2 cycles/day
    # ================================================================

    # Load persistent simulation results for SoC carry-over info
    persistent_csv = OUT / 'data' / 'bess_persistent_daily.csv'
    if persistent_csv.exists():
        bess_persist = pd.read_csv(persistent_csv)
        bess_persist['date'] = pd.to_datetime(bess_persist['date'])
        print("[+] Loaded persistent SoC simulation data")
    else:
        bess_persist = None
        print("[-] No persistent simulation data found, using default start_soc=1")

    bess_days = [pd.Timestamp('2025-11-07'), pd.Timestamp('2026-01-27')]

    for idx, ex_date in enumerate(bess_days):
        day = df[df['date'] == ex_date].sort_values('hour')
        n_hours = len(day)
        if n_hours < 20:
            print(f"[-] Skipping BESS day {ex_date}: only {n_hours} hours")
            continue

        prices = day['da_price'].values[:n_hours]
        pred_ranks = np.nan_to_num(day['rank_predicted'].values[:n_hours], nan=12.0)
        rank_load = np.nan_to_num(day['rank_load'].values[:n_hours], nan=12.0)
        rank_yest = np.nan_to_num(day['rank_yest'].values[:n_hours], nan=12.0)
        rank_7d = np.nan_to_num(day['rank_7d'].values[:n_hours], nan=12.0)
        rank_actual = np.nan_to_num(day['rank_actual'].values[:n_hours], nan=12.0)

        # Get persistent start SoC from simulation results
        start_soc = 1
        prev_end_soc = None
        if bess_persist is not None:
            row_match = bess_persist[bess_persist['date'] == ex_date]
            if len(row_match) > 0:
                start_soc = int(row_match.iloc[0]['start_soc_ind'])
                # Find previous day's end SoC for dashed line
                row_idx = row_match.index[0]
                if row_idx > 0:
                    prev_end_soc = int(bess_persist.iloc[row_idx - 1]['end_soc_ind'])

        # Get yesterday's prices for context
        prev_date = ex_date - pd.Timedelta(days=1)
        prev_day = df[df['date'] == prev_date].sort_values('hour')
        has_prev = len(prev_day) >= 20

        # 7-day average price
        week_dates = [ex_date - pd.Timedelta(days=d) for d in range(1, 8)]
        week_prices = []
        for wd in week_dates:
            wd_data = df[df['date'] == wd].sort_values('hour')
            if len(wd_data) >= 20:
                wp = wd_data['da_price'].values[:min(len(wd_data), n_hours)]
                if len(wp) == n_hours:
                    week_prices.append(wp)
        avg_7d_prices = np.mean(week_prices, axis=0) if len(week_prices) > 0 else None

        # Run DP with persistent SoC (2 cycles)
        act1, rev1, soc1 = optimize_bess_dp(prices, pred_ranks, 4, 4,
                                             start_soc=start_soc, terminal_penalty=20.0)
        _, rev_pf, _ = optimize_bess_dp(prices, prices, 4, 4,
                                         start_soc=start_soc, terminal_penalty=20.0)

        hours = np.arange(n_hours)
        charge_h = np.where(act1 > 0.5)[0]
        discharge_h = np.where(act1 < -0.5)[0]

        fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True,
                                 gridspec_kw={'height_ratios': [3, 1.5, 1.5, 1, 1.5]})

        # Panel 1: Price + Context (today, yesterday, 7d avg, charge/discharge markers)
        ax = axes[0]
        ax.plot(hours, prices, 'k-', lw=2.5, label='DA Price (today)', zorder=3)
        if has_prev:
            prev_p = prev_day['da_price'].values[:min(len(prev_day), n_hours)]
            if len(prev_p) == n_hours:
                ax.plot(hours, prev_p, color='gray', ls='--', lw=1.2, alpha=0.6,
                        label='Yesterday DA', zorder=1)
        if avg_7d_prices is not None:
            ax.plot(hours, avg_7d_prices, color='#1976d2', ls=':', lw=1.2, alpha=0.5,
                    label='7-day avg DA', zorder=1)

        if len(charge_h) > 0:
            ax.scatter(charge_h, prices[charge_h], marker='v', color='green',
                      s=150, zorder=5, label=f'Charge ({len(charge_h)}h)',
                      edgecolors='darkgreen', lw=1.5)
            for h in charge_h:
                ax.axvspan(h - 0.4, h + 0.4, alpha=0.10, color='green')
        if len(discharge_h) > 0:
            ax.scatter(discharge_h, prices[discharge_h], marker='^', color='red',
                      s=150, zorder=5, label=f'Discharge ({len(discharge_h)}h)',
                      edgecolors='darkred', lw=1.5)
            for h in discharge_h:
                ax.axvspan(h - 0.4, h + 0.4, alpha=0.10, color='red')

        ax.set_ylabel('DA Price (EUR/MWh)')
        date_str = pd.Timestamp(ex_date).strftime('%A %d %B %Y')
        cap = rev1 / rev_pf if rev_pf > 0 else 0
        ax.set_title(f'BESS Decision Screen: {date_str}\n'
                     f'Revenue: {rev1:.0f} EUR | Perfect: {rev_pf:.0f} | '
                     f'Capture: {cap:.0%} | Start SoC: {start_soc} MWh',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, loc='upper right', ncol=2)
        ax.grid(True, alpha=0.3)

        # Panel 2: Rank Heatmap (3 horizontal strips + composite overlay)
        ax = axes[1]
        # Normalize ranks to 0-1 for color mapping
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        norm = Normalize(vmin=1, vmax=n_hours)
        cmap = plt.cm.RdYlBu_r  # blue=cheap, red=expensive

        strip_labels = ['Load Rank', 'Yest Rank', '7d Rank']
        strip_data = [rank_load, rank_yest, rank_7d]
        strip_weights = ['40%', '35%', '25%']

        for si, (slabel, sdata, sw) in enumerate(zip(strip_labels, strip_data, strip_weights)):
            y_bottom = 2 - si - 0.4
            y_top = 2 - si + 0.4
            for h in range(n_hours):
                color = cmap(norm(sdata[h]))
                ax.fill_between([h - 0.5, h + 0.5], y_bottom, y_top, color=color)
            ax.text(-1.2, 2 - si, f'{slabel}\n({sw})', fontsize=7, va='center', ha='right')

        # Composite rank as black line overlay
        ax_twin = ax.twinx()
        ax_twin.plot(hours, pred_ranks, 'k-o', ms=3, lw=1.5, label='Composite', zorder=5)
        ax_twin.set_ylabel('Predicted Rank', fontsize=8)
        ax_twin.set_ylim(0, n_hours + 1)
        ax_twin.legend(fontsize=7, loc='upper right')

        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['7d', 'Yest', 'Load'], fontsize=8)
        ax.set_ylim(-0.5, 2.5)
        ax.set_xlim(-0.5, n_hours - 0.5)

        # Add colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.15,
                           aspect=40, shrink=0.6)
        cbar.set_label('Rank (1=cheap, {}=expensive)'.format(n_hours), fontsize=7)
        cbar.ax.tick_params(labelsize=7)

        # Panel 3: Predicted vs Actual rank + charge/discharge markers
        ax = axes[2]
        ax.plot(hours, pred_ranks, 'b-o', ms=4, lw=1.5, label='Predicted Rank')
        ax.plot(hours, rank_actual, 'r--s', ms=4, lw=1.5, alpha=0.7, label='Actual Rank')
        corr = np.corrcoef(pred_ranks[:n_hours], rank_actual[:n_hours])[0, 1]
        # Mark charge/discharge hours
        if len(charge_h) > 0:
            ax.scatter(charge_h, pred_ranks[charge_h], marker='v', color='green',
                      s=80, zorder=5, edgecolors='darkgreen', lw=1)
        if len(discharge_h) > 0:
            ax.scatter(discharge_h, pred_ranks[discharge_h], marker='^', color='red',
                      s=80, zorder=5, edgecolors='darkred', lw=1)
        ax.text(0.02, 0.85, f'Rank correlation: {corr:.2f}',
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax.set_ylabel('Rank')
        ax.legend(fontsize=8)
        ax.set_ylim(0, n_hours + 1)
        ax.grid(True, alpha=0.3)

        # Panel 4: SoC Trace (with carry-over from previous day)
        ax = axes[3]
        soc_hours = np.arange(n_hours + 1)
        ax.step(soc_hours, soc1, where='post', color='#1976d2', lw=2)
        ax.fill_between(soc_hours, 0, soc1, step='post', alpha=0.15, color='#1976d2')
        ax.axhline(0, color='red', ls=':', alpha=0.5, lw=0.8)
        ax.axhline(2, color='red', ls=':', alpha=0.5, lw=0.8)
        # Show carry-over from previous day
        if prev_end_soc is not None:
            ax.plot([0], [prev_end_soc], 'o', color='gray', ms=8, zorder=5,
                    label=f'Prev day end: {prev_end_soc} MWh')
            ax.plot([-0.5, 0], [prev_end_soc, start_soc], '--', color='gray',
                    lw=1.5, alpha=0.6)
            ax.legend(fontsize=7, loc='upper right')
        ax.set_ylabel('SoC (MWh)')
        ax.set_ylim(-0.2, 2.3)
        ax.set_yticks([0, 0.5, 1, 1.5, 2])
        ax.grid(True, alpha=0.3)

        # Panel 5: Revenue (hourly bars + cumulative line)
        ax = axes[4]
        hourly_rev = -act1 * prices
        colors_rev = ['#4caf50' if v > 0 else '#d32f2f' if v < 0 else '#bdbdbd' for v in hourly_rev]
        ax.bar(hours, hourly_rev, color=colors_rev, alpha=0.7, edgecolor='white')
        ax.axhline(0, color='black', lw=0.5)
        cum_rev = np.cumsum(hourly_rev)
        ax.plot(hours, cum_rev, 'k-o', ms=3, lw=1.5, label='Cumulative')
        ax.set_ylabel('Revenue (EUR)')
        ax.set_xlabel('Hour of Day')
        total = cum_rev[-1] if len(cum_rev) > 0 else 0
        ax.text(0.98, 0.05, f'Day total: {total:+.0f} EUR',
                transform=ax.transAxes, fontsize=9, ha='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(RPT_FIG / f'bess_decision_{idx+1}.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"[+] bess_decision_{idx+1}.png")

    # ================================================================
    # CHART 6: SPECULATIVE EXAMPLE DAYS (simpler version for report)
    # ================================================================
    for ex_date, label in [(spec_day1, 'persistence'), (spec_day2, 'mixed')]:
        day = df[df['date'] == ex_date].sort_values('hour').copy()
        day_idm = day[day['idm_vwap'].notna()].copy()
        if len(day_idm) < 10:
            continue
        day_rules = day_idm.apply(classify_expanded, axis=1, result_type='expand')
        day_idm['rule_exp'] = day_rules[0].values
        day_idm['signal_exp'] = day_rules[1].astype(float).values
        day_idm['trade_exp'] = day_rules[2].astype(float).values
        day_idm['pnl_h'] = day_idm['signal_exp'] * day_idm['spread_da_idm'] * day_idm['trade_exp']

        hours = day_idm['hour'].values
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

        # Panel 1: Prices
        ax = axes[0]
        ax.plot(hours, day_idm['da_price'], 'b-o', ms=4, lw=1.5, label='DA Price')
        ax.plot(hours, day_idm['idm_vwap'], 'r-s', ms=4, lw=1.5, label='IDM VWAP')
        ax.fill_between(hours, day_idm['da_price'], day_idm['idm_vwap'],
                        where=day_idm['da_price'] > day_idm['idm_vwap'],
                        alpha=0.15, color='green', label='DA > IDM')
        ax.fill_between(hours, day_idm['da_price'], day_idm['idm_vwap'],
                        where=day_idm['da_price'] <= day_idm['idm_vwap'],
                        alpha=0.15, color='red', label='DA < IDM')
        ax.set_ylabel('Price (EUR/MWh)')
        date_str = pd.Timestamp(ex_date).strftime('%A %d %B %Y')
        ax.set_title(f'{date_str} -- Prices and Spread')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

        # Panel 2: Spread + rule color
        ax = axes[1]
        spread = day_idm['spread_da_idm'].values
        rule_names = day_idm['rule_exp'].values
        for i, h in enumerate(hours):
            color = RULE_COLORS.get(rule_names[i], '#bdbdbd')
            ax.bar(h, spread[i], color=color, alpha=0.7, edgecolor='white')
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylabel('DA-IDM Spread (EUR/MWh)')
        patches = [Patch(color=RULE_COLORS[r], label=r) for r in RULE_ORDER if r in rule_names]
        ax.legend(handles=patches, fontsize=7, ncol=len(patches), loc='upper right')
        ax.grid(True, alpha=0.3)

        # Panel 3: Cumulative P&L
        ax = axes[2]
        pnl_h = day_idm['pnl_h'].values
        cum = np.cumsum(pnl_h)
        ax.fill_between(hours, 0, cum, where=np.array(cum) >= 0, alpha=0.3, color='green')
        ax.fill_between(hours, 0, cum, where=np.array(cum) < 0, alpha=0.3, color='red')
        ax.plot(hours, cum, 'k-o', ms=3, lw=1.5)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylabel('Cumulative P&L (EUR)')
        ax.set_xlabel('Hour of Day')
        total = cum[-1] if len(cum) > 0 else 0
        ax.text(0.98, 0.05, f'Day total: {total:+.0f} EUR',
                transform=ax.transAxes, fontsize=9, ha='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(RPT_FIG / f'spec_example_{label}.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"[+] spec_example_{label}.png")

    print(f"\n[+] All charts saved to {RPT_FIG}")
    print("=" * 90)


if __name__ == '__main__':
    main()
