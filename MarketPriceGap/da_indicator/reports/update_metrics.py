"""
Recalculate speculative strategy metrics using profit factor and expectancy
instead of accuracy. Update charts accordingly.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

BASE = Path(r'C:\Users\20254757\pycharmprojects\ipesoft_eda_data')
OUT  = BASE / 'MarketPriceGap' / 'da_indicator'
RPT_FIG = OUT / 'reports' / 'figures'

sys.path.insert(0, str(OUT / 'scripts'))
from build_da_indicator import load_all_data, build_features, generate_signals


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


def compute_metrics(pnl_series):
    """Compute proper trading metrics from a P&L series."""
    wins = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series < 0]
    flat = pnl_series[pnl_series == 0]

    n = len(pnl_series)
    total_pnl = pnl_series.sum()
    avg_pnl = pnl_series.mean() if n > 0 else 0

    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')

    win_rate = len(wins) / n if n > 0 else 0
    # Expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    return {
        'n': n,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': win_loss_ratio,
        'win_rate': win_rate,
        'expectancy': expectancy,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
    }


def main():
    print("[*] Loading data...")
    df = load_all_data()
    df, daily = build_features(df)
    df = generate_signals(df)
    df['rule'] = df.apply(classify_rule, axis=1)

    mask = df['idm_vwap'].notna() & df['spread_da_idm'].notna() & (df['trade_flag'] > 0)
    traded = df[mask].copy()
    traded['pnl'] = traded['spread_signal'] * traded['spread_da_idm'] * traded['trade_flag']
    traded['year'] = traded['date'].dt.year

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 200})

    # ================================================================
    # PRINT FULL METRICS TABLE
    # ================================================================
    print("\n" + "=" * 90)
    print("  SPECULATIVE STRATEGY - PROPER TRADING METRICS")
    print("=" * 90)

    # Overall
    m = compute_metrics(traded['pnl'])
    print(f"\n--- OVERALL ---")
    print(f"  Trades:         {m['n']}")
    print(f"  Total P&L:      {m['total_pnl']:+,.0f} EUR")
    print(f"  Avg P&L/trade:  {m['avg_pnl']:+.2f} EUR")
    print(f"  Profit Factor:  {m['profit_factor']:.2f}x")
    print(f"  Avg Win:        {m['avg_win']:+.1f} EUR")
    print(f"  Avg Loss:       {m['avg_loss']:.1f} EUR")
    print(f"  Win/Loss Ratio: {m['win_loss_ratio']:.2f}x")
    print(f"  Expectancy:     {m['expectancy']:+.2f} EUR/trade")

    # By year
    print(f"\n--- BY YEAR ---")
    for yr in sorted(traded['year'].unique()):
        sub = traded[traded['year'] == yr]
        m = compute_metrics(sub['pnl'])
        print(f"  {yr}: PF={m['profit_factor']:.2f}x  W/L={m['win_loss_ratio']:.2f}x  "
              f"AvgWin={m['avg_win']:+.1f}  AvgLoss={m['avg_loss']:.1f}  "
              f"Exp={m['expectancy']:+.2f}  Total={m['total_pnl']:+,.0f}")

    # By rule
    print(f"\n--- BY RULE ---")
    rules_order = ['Weekend', 'High Load', 'Persistence', 'Moderate Pers.']
    rule_metrics = []
    for rule in rules_order:
        sub = traded[traded['rule'] == rule]
        if len(sub) > 0:
            m = compute_metrics(sub['pnl'])
            m['rule'] = rule
            rule_metrics.append(m)
            print(f"  {rule:20s}  PF={m['profit_factor']:5.2f}x  W/L={m['win_loss_ratio']:5.2f}x  "
                  f"AvgWin={m['avg_win']:+6.1f}  AvgLoss={m['avg_loss']:6.1f}  "
                  f"Exp={m['expectancy']:+6.2f}  Total={m['total_pnl']:+8,.0f}")

    # By rule AND year
    print(f"\n--- BY RULE AND YEAR ---")
    for rule in rules_order:
        print(f"\n  {rule}:")
        for yr in sorted(traded['year'].unique()):
            sub = traded[(traded['rule'] == rule) & (traded['year'] == yr)]
            if len(sub) > 0:
                m = compute_metrics(sub['pnl'])
                print(f"    {yr}: PF={m['profit_factor']:5.2f}x  W/L={m['win_loss_ratio']:5.2f}x  "
                      f"AvgWin={m['avg_win']:+6.1f}  AvgLoss={m['avg_loss']:6.1f}  "
                      f"Exp={m['expectancy']:+6.2f}  n={m['n']}")

    # ================================================================
    # UPDATED CHARTS
    # ================================================================

    rm = pd.DataFrame(rule_metrics)

    # --- Chart: Rule breakdown with profit factor and win/loss ratio ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel 1: Profit Factor
    ax = axes[0]
    colors = ['#4caf50' if v > 1.1 else '#ff9800' if v > 1.0 else '#d32f2f' for v in rm['profit_factor']]
    bars = ax.barh(rm['rule'], rm['profit_factor'], color=colors, edgecolor='white')
    ax.axvline(1.0, color='red', ls='--', alpha=0.5, label='Breakeven')
    ax.set_xlabel('Profit Factor (Gross Profit / Gross Loss)')
    ax.set_title('Profit Factor by Rule')
    for i, row in rm.iterrows():
        pf = row['profit_factor']
        ax.text(pf + 0.03, i, f"{pf:.2f}x", va='center', fontsize=9, fontweight='bold')
    ax.set_xlim(0, max(rm['profit_factor']) * 1.25)
    ax.legend(fontsize=8)

    # Panel 2: Win/Loss Ratio (avg win size vs avg loss size)
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
                f"Ratio: {row['win_loss_ratio']:.1f}x", va='center', fontsize=8)

    # Panel 3: Expectancy (EUR per trade)
    ax = axes[2]
    colors = ['#4caf50' if v > 0 else '#d32f2f' for v in rm['expectancy']]
    ax.barh(rm['rule'], rm['expectancy'], color=colors, edgecolor='white')
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel('Expectancy (EUR/trade)')
    ax.set_title('Expected Value per Trade')
    for i, row in rm.iterrows():
        side = 0.3 if row['expectancy'] >= 0 else -0.3
        ha = 'left' if row['expectancy'] >= 0 else 'right'
        ax.text(row['expectancy'] + side, i,
                f"{row['expectancy']:+.1f} (n={row['n']})", va='center', fontsize=9, ha=ha)

    fig.suptitle('Which Rules Make Money? (Profit Factor, Win/Loss Ratio, Expectancy)',
                 fontweight='bold')
    fig.tight_layout()
    fig.savefig(RPT_FIG / 'spec_rule_breakdown.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[+] spec_rule_breakdown.png UPDATED")

    # --- Chart: Rolling Profit Factor (instead of rolling accuracy) ---
    daily_pnl = traded.groupby('date').agg(
        pnl=('pnl', 'sum'),
        gross_profit=('pnl', lambda x: x[x > 0].sum()),
        gross_loss=('pnl', lambda x: abs(x[x < 0].sum())),
    ).reset_index()
    daily_pnl['date'] = pd.to_datetime(daily_pnl['date'])
    daily_pnl = daily_pnl.sort_values('date')

    # 30-day rolling profit factor
    daily_pnl['roll_gp'] = daily_pnl['gross_profit'].rolling(30, min_periods=10).sum()
    daily_pnl['roll_gl'] = daily_pnl['gross_loss'].rolling(30, min_periods=10).sum()
    daily_pnl['roll_pf'] = daily_pnl['roll_gp'] / daily_pnl['roll_gl'].clip(lower=1)
    # 30-day rolling avg P&L per day
    daily_pnl['roll_avg_pnl'] = daily_pnl['pnl'].rolling(30, min_periods=10).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Rolling Profit Factor
    ax1.plot(daily_pnl['date'], daily_pnl['roll_pf'], 'b-', lw=1.5)
    ax1.axhline(1.0, color='red', ls='--', alpha=0.5, label='Breakeven (PF=1.0)')
    ax1.axvline(pd.Timestamp('2026-01-01'), color='red', ls='--', alpha=0.3)
    ax1.axvspan(pd.Timestamp('2026-01-01'), daily_pnl['date'].max(), alpha=0.08, color='red')
    ax1.set_ylabel('30-day Rolling Profit Factor')
    ax1.set_title('Signal Quality Over Time: Profit Factor and Daily P\\&L')
    ax1.set_ylim(0, 5)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Rolling avg daily P&L
    ax2.fill_between(daily_pnl['date'], 0, daily_pnl['roll_avg_pnl'],
                     where=daily_pnl['roll_avg_pnl'] >= 0, alpha=0.3, color='green')
    ax2.fill_between(daily_pnl['date'], 0, daily_pnl['roll_avg_pnl'],
                     where=daily_pnl['roll_avg_pnl'] < 0, alpha=0.3, color='red')
    ax2.plot(daily_pnl['date'], daily_pnl['roll_avg_pnl'], 'k-', lw=1)
    ax2.axhline(0, color='black', lw=0.5)
    ax2.axvline(pd.Timestamp('2026-01-01'), color='red', ls='--', alpha=0.3)
    ax2.set_ylabel('30-day Rolling Avg Daily P&L (EUR)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RPT_FIG / 'spec_rolling_pf.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("[+] spec_rolling_pf.png")

    # --- Chart: P&L with Jan 2026 focus (updated annotations) ---
    daily_spec = traded.groupby('date').agg(
        pnl=('pnl', 'sum'), n=('pnl', 'count'),
    ).reset_index()
    daily_spec['date'] = pd.to_datetime(daily_spec['date'])
    daily_spec = daily_spec.sort_values('date')
    daily_spec['cum_pnl'] = daily_spec['pnl'].cumsum()

    import matplotlib.dates as mdates
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    ax1.plot(daily_spec['date'], daily_spec['cum_pnl'], 'b-', lw=1.2)
    ax1.axvspan(pd.Timestamp('2026-01-01'), daily_spec['date'].max(), alpha=0.08, color='red')
    ax1.axvline(pd.Timestamp('2026-01-01'), color='red', ls='--', alpha=0.4, label='2026 start')
    ax1.set_ylabel('Cumulative P&L (EUR)')
    ax1.set_title('Speculative Strategy: Cumulative P&L')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    jan_data = daily_spec[(daily_spec['date'] >= '2025-12-01') & (daily_spec['date'] <= '2026-02-05')]
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
        jan26 = jan_data[jan_data['date'] >= '2026-01-01']
        m_jan = compute_metrics(traded[traded['year'] == 2026]['pnl'])
        ax2.set_title(f'Dec 2025 -- Feb 2026  |  Jan 2026: {m_jan["total_pnl"]:+,.0f} EUR, '
                      f'PF={m_jan["profit_factor"]:.1f}x, Exp={m_jan["expectancy"]:+.1f} EUR/trade')
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RPT_FIG / 'spec_pnl_jan2026.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("[+] spec_pnl_jan2026.png UPDATED")

    print("\n" + "=" * 90)


if __name__ == '__main__':
    main()
