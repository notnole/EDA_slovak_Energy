"""
Generate all charts for the BESS and Speculative trader report PDFs.
Each chart is saved as a high-quality PNG for LaTeX inclusion.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import sys

BASE = Path(r'C:\Users\20254757\pycharmprojects\ipesoft_eda_data')
OUT  = BASE / 'MarketPriceGap' / 'da_indicator'
RPT  = OUT / 'reports'
RPT_FIG = RPT / 'figures'
RPT_FIG.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(OUT / 'scripts'))
from build_da_indicator import load_all_data, build_features, generate_signals

# Replicate rule classification from rule_breakdown.py
def classify_rule(row):
    is_wknd = bool(row.get('is_weekend', False))
    load_dev = row.get('load_fcst_dev', 0)
    spread_yest = row.get('spread_yest_h', 0)
    spread_yest_daily = row.get('spread_mean_yest', 0)
    if pd.isna(load_dev): load_dev = 0
    if pd.isna(spread_yest): spread_yest = 0
    if pd.isna(spread_yest_daily): spread_yest_daily = 0
    if is_wknd: return 'Weekend'
    if load_dev > 85: return 'High Load'
    if abs(spread_yest) >= 20: return 'Persistence'
    if abs(spread_yest_daily) >= 10: return 'Moderate Pers.'
    return 'Baseline'


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
    print("[*] Loading data for report charts...")
    df = load_all_data()
    df, daily = build_features(df)
    df = generate_signals(df)
    df['rule'] = df.apply(classify_rule, axis=1)

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 200})

    # ================================================================
    # FIND GOOD EXAMPLE DAYS (January 2026 preferred)
    # ================================================================
    jan26 = df[(df['date'] >= '2026-01-01') & (df['date'] <= '2026-01-31')].copy()
    jan26_dates = sorted(jan26['date'].unique())

    # For speculative: find a weekend and a weekday in Jan 2026 with IDM data
    spec_examples = []
    for d in jan26_dates:
        day = jan26[jan26['date'] == d]
        if day['idm_vwap'].notna().sum() >= 20:
            spread = day['spread_da_idm'].dropna()
            if len(spread) > 0:
                is_wknd = day['is_weekend'].iloc[0]
                spec_examples.append({
                    'date': d, 'is_weekend': is_wknd,
                    'mean_spread': spread.mean(), 'n_hours': len(spread)
                })

    spec_ex_df = pd.DataFrame(spec_examples)
    # Pick one weekend, one weekday
    wknd_days = spec_ex_df[spec_ex_df['is_weekend'] == True].sort_values('mean_spread', ascending=False)
    wkdy_days = spec_ex_df[spec_ex_df['is_weekend'] == False].sort_values('mean_spread', key=abs, ascending=False)

    spec_day1 = wknd_days.iloc[0]['date'] if len(wknd_days) > 0 else jan26_dates[0]
    spec_day2 = wkdy_days.iloc[0]['date'] if len(wkdy_days) > 0 else jan26_dates[-1]
    print(f"[+] Speculative example days: {spec_day1} (weekend), {spec_day2} (weekday)")

    # For BESS: use handpicked days with recognisable price shapes
    # Day 1: classical two-peak profile from Nov 2025 (night low, morning+evening peaks)
    # Day 2: best-structured Jan 2026 day (morning peak h09, mild evening bump)
    bess_day1 = pd.Timestamp('2025-11-07')
    bess_day2 = pd.Timestamp('2026-01-27')
    print(f"[+] BESS example days: {bess_day1.date()} (classical shape), {bess_day2.date()} (Jan 2026)")

    # ================================================================
    # SPECULATIVE CHARTS
    # ================================================================

    # --- S1: Example day - Weekend ---
    for idx, (ex_date, label) in enumerate([(spec_day1, 'weekend'), (spec_day2, 'weekday')]):
        day = df[df['date'] == ex_date].sort_values('hour').copy()
        day_traded = day[day['idm_vwap'].notna()].copy()

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        hours = day_traded['hour'].values

        # Panel 1: DA and IDM prices
        ax = axes[0]
        ax.plot(hours, day_traded['da_price'], 'b-o', ms=4, lw=1.5, label='DA Price')
        ax.plot(hours, day_traded['idm_vwap'], 'r-s', ms=4, lw=1.5, label='IDM VWAP')
        ax.fill_between(hours, day_traded['da_price'], day_traded['idm_vwap'],
                        where=day_traded['da_price'] > day_traded['idm_vwap'],
                        alpha=0.15, color='green', label='DA > IDM (sell DA profits)')
        ax.fill_between(hours, day_traded['da_price'], day_traded['idm_vwap'],
                        where=day_traded['da_price'] <= day_traded['idm_vwap'],
                        alpha=0.15, color='red', label='DA < IDM (buy DA profits)')
        ax.set_ylabel('Price (EUR/MWh)')
        ax.set_title(f'{pd.Timestamp(ex_date).strftime("%A %d %B %Y")} -- Prices and Spread')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

        # Panel 2: Spread and signal
        ax = axes[1]
        spread = day_traded['spread_da_idm'].values
        colors = ['green' if s > 0 else 'red' for s in spread]
        ax.bar(hours, spread, color=colors, alpha=0.6, edgecolor='white')
        ax.axhline(0, color='black', lw=0.5)

        # Show signal arrows
        for _, r in day_traded.iterrows():
            if r['trade_flag'] > 0:
                color = 'green' if r['spread_signal'] > 0 else 'red'
                marker = '^' if r['spread_signal'] > 0 else 'v'
                y_pos = spread.max() * 1.1 if r['spread_signal'] > 0 else spread.min() * 1.1
                ax.scatter(r['hour'], y_pos, marker=marker, color=color, s=60, zorder=5)
        ax.set_ylabel('DA-IDM Spread (EUR/MWh)')

        # Annotate the rule
        rules = day_traded['rule'].unique()
        rule_str = ', '.join(rules)
        ax.text(0.02, 0.95, f'Signal rule: {rule_str}',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax.grid(True, alpha=0.3)

        # Panel 3: Cumulative P&L through the day
        ax = axes[2]
        day_traded['pnl_hour'] = day_traded['spread_signal'] * day_traded['spread_da_idm'] * day_traded['trade_flag']
        cum_pnl = day_traded['pnl_hour'].cumsum().values
        ax.fill_between(hours, 0, cum_pnl,
                        where=np.array(cum_pnl) >= 0, alpha=0.3, color='green')
        ax.fill_between(hours, 0, cum_pnl,
                        where=np.array(cum_pnl) < 0, alpha=0.3, color='red')
        ax.plot(hours, cum_pnl, 'k-o', ms=3, lw=1.5)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylabel('Cumulative P&L (EUR)')
        ax.set_xlabel('Hour of Day')
        total = cum_pnl[-1] if len(cum_pnl) > 0 else 0
        correct = (day_traded['pnl_hour'] > 0).sum()
        total_h = len(day_traded[day_traded['trade_flag'] > 0])
        ax.text(0.98, 0.05, f'Day total: {total:+.0f} EUR  |  {correct}/{total_h} correct',
                transform=ax.transAxes, fontsize=9, ha='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(RPT_FIG / f'spec_example_{label}.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"[+] spec_example_{label}.png")

    # --- S3: P&L focused on Jan 2026 ---
    mask = df['idm_vwap'].notna() & df['spread_da_idm'].notna() & (df['trade_flag'] > 0)
    traded = df[mask].copy()
    traded['pnl'] = traded['spread_signal'] * traded['spread_da_idm'] * traded['trade_flag']
    traded['correct'] = ((traded['spread_signal'] > 0) == (traded['spread_da_idm'] > 0)).astype(int)

    daily_spec = traded.groupby('date').agg(
        pnl=('pnl', 'sum'), n_correct=('correct', 'sum'),
        n_traded=('correct', 'count'),
    ).reset_index()
    daily_spec['date'] = pd.to_datetime(daily_spec['date'])
    daily_spec = daily_spec.sort_values('date')
    daily_spec['cum_pnl'] = daily_spec['pnl'].cumsum()
    daily_spec['accuracy'] = daily_spec['n_correct'] / daily_spec['n_traded']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
    jan26_start = pd.Timestamp('2025-12-01')
    jan26_end = pd.Timestamp('2026-02-05')

    # Full history (top)
    ax1.plot(daily_spec['date'], daily_spec['cum_pnl'], 'b-', lw=1.2)
    ax1.axvspan(pd.Timestamp('2026-01-01'), jan26_end, alpha=0.08, color='red')
    ax1.axvline(pd.Timestamp('2026-01-01'), color='red', ls='--', alpha=0.4, label='2026 start')
    ax1.set_ylabel('Cumulative P\\&L (EUR)')
    ax1.set_title('Speculative Strategy: Full Period and January 2026 Focus')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Jan 2026 zoom (bottom)
    jan_data = daily_spec[(daily_spec['date'] >= jan26_start) & (daily_spec['date'] <= jan26_end)]
    if len(jan_data) > 0:
        # Reset cumulative for this period
        jan_cum = jan_data['pnl'].cumsum()
        colors_bar = ['green' if v > 0 else 'red' for v in jan_data['pnl'].values]
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
        total_jan = jan_data[jan_data['date'] >= '2026-01-01']['pnl'].sum()
        acc_jan = jan_data[jan_data['date'] >= '2026-01-01']['n_correct'].sum() / jan_data[jan_data['date'] >= '2026-01-01']['n_traded'].sum()
        ax2.set_title(f'December 2025 -- February 2026  |  Jan 2026: {total_jan:+,.0f} EUR, {acc_jan:.0%} accuracy')
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RPT_FIG / 'spec_pnl_jan2026.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("[+] spec_pnl_jan2026.png")

    # --- S4: Rule breakdown ---
    rules_order = ['Weekend', 'High Load', 'Persistence', 'Moderate Pers.']
    rule_stats = []
    for rule in rules_order:
        sub = traded[traded['rule'] == rule]
        if len(sub) > 0:
            rule_stats.append({
                'rule': rule, 'accuracy': sub['correct'].mean(),
                'avg_pnl': sub['pnl'].mean(), 'total_pnl': sub['pnl'].sum(),
                'n': len(sub),
            })
    rd = pd.DataFrame(rule_stats)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    colors_acc = ['#4caf50' if v > 0.55 else '#ff9800' if v > 0.50 else '#d32f2f' for v in rd['accuracy']]
    ax1.barh(rd['rule'], rd['accuracy'], color=colors_acc, edgecolor='white')
    ax1.axvline(0.5, color='red', ls='--', alpha=0.5, label='Random (50%)')
    ax1.set_xlabel('Accuracy')
    ax1.set_title('Direction Accuracy by Rule')
    for i, row in rd.iterrows():
        ax1.text(row['accuracy'] + 0.01, i, f"{row['accuracy']:.1%} (n={row['n']})", va='center', fontsize=9)
    ax1.set_xlim(0, 0.85)
    ax1.legend(fontsize=8)

    colors_pnl = ['#4caf50' if v > 0 else '#d32f2f' for v in rd['total_pnl']]
    ax2.barh(rd['rule'], rd['total_pnl'] / 1000, color=colors_pnl, edgecolor='white')
    ax2.axvline(0, color='black', lw=0.5)
    ax2.set_xlabel('Total P&L (kEUR)')
    ax2.set_title('Total P&L Contribution by Rule')
    for i, row in rd.iterrows():
        side = row['total_pnl'] / 1000
        ax2.text(max(side + 0.3, 0.3), i, f"{row['total_pnl']/1000:+.1f}k", va='center', fontsize=9)
    fig.suptitle('Which Rules Drive Performance?', fontweight='bold')
    fig.tight_layout()
    fig.savefig(RPT_FIG / 'spec_rule_breakdown.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("[+] spec_rule_breakdown.png")

    # --- S5: Rolling accuracy with 2026 ---
    daily_acc = traded.groupby('date').agg(accuracy=('correct', 'mean')).reset_index()
    daily_acc['date'] = pd.to_datetime(daily_acc['date'])
    daily_acc = daily_acc.sort_values('date')
    daily_acc['rolling'] = daily_acc['accuracy'].rolling(30, min_periods=10).mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily_acc['date'], daily_acc['rolling'], 'b-', lw=1.5, label='30-day rolling accuracy')
    ax.axhline(0.5, color='red', ls='--', alpha=0.5, label='Random (50%)')
    ax.axvline(pd.Timestamp('2026-01-01'), color='red', ls='--', alpha=0.3)
    ax.axvspan(pd.Timestamp('2026-01-01'), daily_acc['date'].max(), alpha=0.08, color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Accuracy')
    ax.set_title('Signal Accuracy Over Time -- Does It Persist Into 2026?')
    ax.set_ylim(0.35, 0.85)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RPT_FIG / 'spec_rolling_accuracy.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("[+] spec_rolling_accuracy.png")

    # ================================================================
    # BESS CHARTS (2 cycles/day, persistent SoC)
    # ================================================================

    # Load persistent simulation results
    persistent_csv = OUT / 'data' / 'bess_persistent_daily.csv'
    if persistent_csv.exists():
        bess_persist = pd.read_csv(persistent_csv)
        bess_persist['date'] = pd.to_datetime(bess_persist['date'])
        print("[+] Loaded persistent SoC simulation data")
    else:
        bess_persist = None
        print("[-] No persistent simulation data, using default start_soc=1")

    # --- B1/B2: Example days (2 cycles, persistent SoC, 5-panel decision screen) ---
    for idx, ex_date in enumerate([bess_day1, bess_day2]):
        day = df[df['date'] == ex_date].sort_values('hour')
        n_hours = len(day)
        if n_hours < 20:
            continue
        prices = day['da_price'].values[:n_hours]
        pred_ranks = np.nan_to_num(day['rank_predicted'].values[:n_hours], nan=12.0)
        rank_load = np.nan_to_num(day['rank_load'].values[:n_hours], nan=12.0)
        rank_yest = np.nan_to_num(day['rank_yest'].values[:n_hours], nan=12.0)
        rank_7d = np.nan_to_num(day['rank_7d'].values[:n_hours], nan=12.0)
        rank_actual = np.nan_to_num(day['rank_actual'].values[:n_hours], nan=12.0)

        # Get persistent start SoC
        start_soc = 1
        prev_end_soc = None
        if bess_persist is not None:
            row_match = bess_persist[bess_persist['date'] == ex_date]
            if len(row_match) > 0:
                start_soc = int(row_match.iloc[0]['start_soc_ind'])
                row_idx = row_match.index[0]
                if row_idx > 0:
                    prev_end_soc = int(bess_persist.iloc[row_idx - 1]['end_soc_ind'])

        # Yesterday's and 7d average prices for context
        prev_date = ex_date - pd.Timedelta(days=1)
        prev_day = df[df['date'] == prev_date].sort_values('hour')
        has_prev = len(prev_day) >= 20

        week_dates = [ex_date - pd.Timedelta(days=d) for d in range(1, 8)]
        week_prices = []
        for wd in week_dates:
            wd_data = df[df['date'] == wd].sort_values('hour')
            if len(wd_data) >= 20:
                wp = wd_data['da_price'].values[:min(len(wd_data), n_hours)]
                if len(wp) == n_hours:
                    week_prices.append(wp)
        avg_7d_prices = np.mean(week_prices, axis=0) if len(week_prices) > 0 else None

        # 2 cycles with persistent SoC
        act, rev, soc = optimize_bess_dp(prices, pred_ranks, 4, 4,
                                          start_soc=start_soc, terminal_penalty=20.0)
        _, rev_pf, _ = optimize_bess_dp(prices, prices, 4, 4,
                                         start_soc=start_soc, terminal_penalty=20.0)

        hours = np.arange(n_hours)
        charge_h = np.where(act > 0.5)[0]
        discharge_h = np.where(act < -0.5)[0]

        # 5-panel decision screen
        fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True,
                                 gridspec_kw={'height_ratios': [3, 1.5, 1.5, 1, 1.5]})

        # Panel 1: Price + Context
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
        cap = rev / rev_pf if rev_pf > 0 else 0
        ax.set_title(f'{date_str} -- 2 Cycles: Revenue {rev:.0f} EUR '
                     f'(Perfect: {rev_pf:.0f}, Capture: {cap:.0%}, Start SoC: {start_soc})',
                     fontsize=10)
        ax.legend(fontsize=7, loc='upper right', ncol=2)
        ax.grid(True, alpha=0.3)

        # Panel 2: Rank Heatmap
        ax = axes[1]
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        norm = Normalize(vmin=1, vmax=n_hours)
        cmap = plt.cm.RdYlBu_r

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

        ax_twin = ax.twinx()
        ax_twin.plot(hours, pred_ranks, 'k-o', ms=3, lw=1.5, label='Composite', zorder=5)
        ax_twin.set_ylabel('Predicted Rank', fontsize=8)
        ax_twin.set_ylim(0, n_hours + 1)
        ax_twin.legend(fontsize=7, loc='upper right')

        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['7d', 'Yest', 'Load'], fontsize=8)
        ax.set_ylim(-0.5, 2.5)
        ax.set_xlim(-0.5, n_hours - 0.5)

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.15,
                           aspect=40, shrink=0.6)
        cbar.set_label('Rank (1=cheap, {}=expensive)'.format(n_hours), fontsize=7)
        cbar.ax.tick_params(labelsize=7)

        # Panel 3: Predicted vs Actual rank
        ax = axes[2]
        ax.plot(hours, pred_ranks, 'b-o', ms=4, lw=1.5, label='Predicted Rank')
        ax.plot(hours, rank_actual, 'r--s', ms=4, lw=1.5, alpha=0.7, label='Actual Rank')
        corr = np.corrcoef(pred_ranks[:n_hours], rank_actual[:n_hours])[0, 1]
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

        # Panel 4: SoC Trace
        ax = axes[3]
        soc_hours = np.arange(n_hours + 1)
        ax.step(soc_hours, soc, where='post', color='#1976d2', lw=2)
        ax.fill_between(soc_hours, 0, soc, step='post', alpha=0.15, color='#1976d2')
        ax.axhline(0, color='red', ls=':', alpha=0.5, lw=0.8)
        ax.axhline(2, color='red', ls=':', alpha=0.5, lw=0.8)
        if prev_end_soc is not None:
            ax.plot([0], [prev_end_soc], 'o', color='gray', ms=8, zorder=5,
                    label=f'Prev day end: {prev_end_soc} MWh')
            ax.plot([-0.5, 0], [prev_end_soc, start_soc], '--', color='gray',
                    lw=1.5, alpha=0.6)
            ax.legend(fontsize=7, loc='upper right')
        ax.set_ylabel('SoC (MWh)')
        ax.set_ylim(-0.2, 2.3)
        ax.set_yticks([0, 0.5, 1, 1.5, 2])
        ax.text(0.02, 0.85, 'Capacity: 2 MWh', transform=ax.transAxes,
                fontsize=8, color='gray')
        ax.grid(True, alpha=0.3)

        # Panel 5: Revenue
        ax = axes[4]
        hourly_rev = -act * prices
        colors_h = ['#4caf50' if v > 0 else '#d32f2f' if v < 0 else '#bdbdbd' for v in hourly_rev]
        ax.bar(hours, hourly_rev, color=colors_h, alpha=0.7, edgecolor='white')
        ax.axhline(0, color='black', lw=0.5)
        cum = np.cumsum(hourly_rev)
        ax.plot(hours, cum, 'k--o', ms=3, lw=1.5, label='Cumulative')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Revenue (EUR)')
        total = cum[-1] if len(cum) > 0 else 0
        ax.text(0.98, 0.05, f'Day total: {total:+.0f} EUR',
                transform=ax.transAxes, fontsize=9, ha='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(RPT_FIG / f'bess_example_{idx+1}_2cycle.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"[+] bess_example_{idx+1}_2cycle.png")

    # --- B3: BESS P&L with Jan 2026 focus (from persistent simulation) ---
    bess_daily_csv = OUT / 'data' / 'bess_persistent_daily.csv'
    if not bess_daily_csv.exists():
        bess_daily_csv = OUT / 'data' / 'bess_daily.csv'
    bess_daily = pd.read_csv(bess_daily_csv)
    bess_daily['date'] = pd.to_datetime(bess_daily['date'])
    bess_daily = bess_daily.sort_values('date')

    # Determine column names (persistent vs old format)
    rev_ind_col = 'rev_indicator' if 'rev_indicator' in bess_daily.columns else 'rev_ind'
    rev_pf_col = 'rev_perfect' if 'rev_perfect' in bess_daily.columns else 'rev_pf'
    rev_naive_col = 'rev_naive'
    cap_col = 'capture_rate' if 'capture_rate' in bess_daily.columns else 'cap'

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
    ax1.plot(bess_daily['date'], bess_daily[rev_ind_col].cumsum(), 'b-', lw=1.2,
             label='Indicator (2 cycles)')
    ax1.plot(bess_daily['date'], bess_daily[rev_pf_col].cumsum(), 'g--', lw=0.8, alpha=0.6,
             label='Perfect foresight')
    ax1.plot(bess_daily['date'], bess_daily[rev_naive_col].cumsum(), 'r:', lw=0.8, alpha=0.6,
             label='Naive (fixed hours)')
    ax1.axvspan(pd.Timestamp('2026-01-01'), bess_daily['date'].max(), alpha=0.08, color='red')
    ax1.axvline(pd.Timestamp('2026-01-01'), color='red', ls='--', alpha=0.4)
    ax1.set_ylabel('Cumulative Revenue (EUR)')
    ax1.set_title('BESS Revenue: 2 Cycles/Day, Persistent SoC')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Jan 2026 zoom
    jan_bess = bess_daily[(bess_daily['date'] >= '2025-12-01') & (bess_daily['date'] <= '2026-02-05')]
    if len(jan_bess) > 0:
        ax2.bar(jan_bess['date'], jan_bess[rev_ind_col], color='#1976d2', alpha=0.7,
                width=0.8, label='Indicator')
        ax2.bar(jan_bess['date'], jan_bess[rev_naive_col], color='#ff9800', alpha=0.5,
                width=0.4, label='Naive')
        ax2.axhline(0, color='black', lw=0.5)
        ax2.axvline(pd.Timestamp('2026-01-01'), color='red', ls='--', alpha=0.4)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        ax2.tick_params(axis='x', rotation=45)
        jan_only = jan_bess[jan_bess['date'] >= '2026-01-01']
        tot_ind = jan_only[rev_ind_col].sum()
        tot_naive = jan_only[rev_naive_col].sum()
        ax2.set_title(f'Dec 2025 -- Feb 2026  |  Jan 2026 Indicator: {tot_ind:+,.0f} EUR, '
                      f'Naive: {tot_naive:+,.0f} EUR')
    ax2.set_ylabel('Daily Revenue (EUR)')
    ax2.set_xlabel('Date')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RPT_FIG / 'bess_pnl_jan2026.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("[+] bess_pnl_jan2026.png")

    # --- B4: Capture rate ---
    fig, ax = plt.subplots(figsize=(8, 4))
    cr = bess_daily[cap_col].dropna()
    ax.hist(cr, bins=40, color='#1976d2', edgecolor='white', alpha=0.8)
    ax.axvline(cr.median(), color='red', ls='--', label=f'Median: {cr.median():.1%}')
    ax.axvline(cr.mean(), color='orange', ls='--', label=f'Mean: {cr.mean():.1%}')
    ax.set_xlabel('Capture Rate (Indicator / Perfect Foresight)')
    ax.set_ylabel('Number of Days')
    ax.set_title('How Close to Optimal? Daily Capture Rate (2 Cycles, Persistent SoC)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(RPT_FIG / 'bess_capture_rate.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("[+] bess_capture_rate.png")

    # --- B5: Monthly revenue ---
    monthly_csv = OUT / 'data' / 'bess_persistent_monthly.csv'
    if not monthly_csv.exists():
        monthly_csv = OUT / 'data' / 'bess_monthly.csv'
    bess_monthly = pd.read_csv(monthly_csv)

    # Determine column names
    m_ind_col = 'rev_ind' if 'rev_ind' in bess_monthly.columns else 'rev_indicator'
    m_pf_col = 'rev_pf' if 'rev_pf' in bess_monthly.columns else 'rev_perfect'
    m_naive_col = 'rev_naive'
    m_month_col = 'month' if 'month' in bess_monthly.columns else 'month_period'

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(bess_monthly))
    w = 0.25
    ax.bar([i - w for i in x], bess_monthly[m_ind_col], w, label='Indicator', color='#1976d2')
    ax.bar(x, bess_monthly[m_pf_col], w, label='Perfect foresight', color='#4caf50', alpha=0.7)
    ax.bar([i + w for i in x], bess_monthly[m_naive_col], w, label='Naive', color='#ff9800')
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(m) for m in bess_monthly[m_month_col]], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Monthly Revenue (EUR)')
    ax.set_title('BESS Monthly Revenue: 2 Cycles/Day, Persistent SoC')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RPT_FIG / 'bess_monthly_revenue.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("[+] bess_monthly_revenue.png")

    print(f"\n[+] All report charts saved to {RPT_FIG}")


if __name__ == '__main__':
    main()
