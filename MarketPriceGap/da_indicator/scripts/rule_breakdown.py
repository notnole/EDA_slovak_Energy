"""
Deeper analysis: Break down speculative trading performance by signal rule.
Identifies which of the 5 rules drives accuracy and P&L.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(r'C:\Users\20254757\pycharmprojects\ipesoft_eda_data')
OUT  = BASE / 'MarketPriceGap' / 'da_indicator'


def classify_rule(row):
    """Replicate the rule logic to tag each hour with its triggering rule."""
    is_wknd = bool(row.get('is_weekend', False))
    load_dev = row.get('load_fcst_dev', 0)
    spread_yest = row.get('spread_yest_h', 0)
    spread_yest_daily = row.get('spread_mean_yest', 0)

    if pd.isna(load_dev): load_dev = 0
    if pd.isna(spread_yest): spread_yest = 0
    if pd.isna(spread_yest_daily): spread_yest_daily = 0

    if is_wknd:
        return 'R1: Weekend'
    if load_dev > 85:
        return 'R2: High Load Dev'
    if abs(spread_yest) >= 20:
        return 'R3: Large Persistence'
    if abs(spread_yest_daily) >= 10:
        return 'R4: Moderate Persistence'
    return 'R5: Baseline'


def main():
    # Load the indicator signals
    sig = pd.read_csv(OUT / 'data' / 'indicator_signals.csv')
    sig['timestamp_hour'] = pd.to_datetime(sig['timestamp_hour'])
    sig['date'] = pd.to_datetime(sig['date'])

    # We need the extra features for rule classification - reload from main script
    import sys
    sys.path.insert(0, str(OUT / 'scripts'))
    from build_da_indicator import load_all_data, build_features, generate_signals

    df = load_all_data()
    df, daily = build_features(df)
    df = generate_signals(df)

    # Classify each hour by rule
    df['rule'] = df.apply(classify_rule, axis=1)

    # Filter to traded hours with IDM data
    mask = df['idm_vwap'].notna() & df['spread_da_idm'].notna() & (df['trade_flag'] > 0)
    traded = df[mask].copy()
    traded['pnl'] = traded['spread_signal'] * traded['spread_da_idm'] * traded['trade_flag']
    traded['correct'] = ((traded['spread_signal'] > 0) == (traded['spread_da_idm'] > 0)).astype(int)
    traded['year'] = traded['date'].dt.year

    print("=" * 70)
    print("  SIGNAL RULE BREAKDOWN")
    print("=" * 70)

    # Overall by rule
    print("\n--- OVERALL BY RULE ---")
    for rule in ['R1: Weekend', 'R2: High Load Dev', 'R3: Large Persistence',
                 'R4: Moderate Persistence', 'R5: Baseline']:
        sub = traded[traded['rule'] == rule]
        if len(sub) > 0:
            acc = sub['correct'].mean()
            avg_pnl = sub['pnl'].mean()
            total = sub['pnl'].sum()
            pct = len(sub) / len(traded) * 100
            print(f"  {rule:30s}  n={len(sub):5d} ({pct:4.1f}%)  "
                  f"Acc={acc:.1%}  Avg={avg_pnl:+.2f}  Total={total:+,.0f} EUR")

    # By rule AND year
    print("\n--- BY RULE AND YEAR ---")
    for rule in ['R1: Weekend', 'R2: High Load Dev', 'R3: Large Persistence',
                 'R4: Moderate Persistence', 'R5: Baseline']:
        print(f"\n  {rule}:")
        for yr in sorted(traded['year'].unique()):
            sub = traded[(traded['rule'] == rule) & (traded['year'] == yr)]
            if len(sub) > 0:
                acc = sub['correct'].mean()
                avg_pnl = sub['pnl'].mean()
                total = sub['pnl'].sum()
                print(f"    {yr}: n={len(sub):4d}  Acc={acc:.1%}  "
                      f"Avg={avg_pnl:+.2f}  Total={total:+,.0f} EUR")

    # R5 Baseline is no-trade - show what would happen if traded
    no_trade = df[df['idm_vwap'].notna() & df['spread_da_idm'].notna() &
                  (df['trade_flag'] == 0)].copy()
    if len(no_trade) > 0:
        no_trade['pnl_hypothetical'] = no_trade['spread_signal'] * no_trade['spread_da_idm']
        no_trade['correct'] = ((no_trade['spread_signal'] > 0) == (no_trade['spread_da_idm'] > 0)).astype(int)
        print(f"\n--- R5 BASELINE (NOT TRADED - HYPOTHETICAL) ---")
        print(f"  n={len(no_trade)}, Acc={no_trade['correct'].mean():.1%}, "
              f"Avg P&L={no_trade['pnl_hypothetical'].mean():.2f}, "
              f"Total={no_trade['pnl_hypothetical'].sum():+,.0f} EUR")

    # BESS: analyze hour selection accuracy
    print("\n\n--- BESS HOUR SELECTION ANALYSIS ---")
    bess = pd.read_csv(OUT / 'data' / 'bess_daily.csv')
    bess['date'] = pd.to_datetime(bess['date'])

    # Parse charge/discharge hours from string lists (may contain np.int64)
    import ast
    import re as re2
    def parse_hour_list(s):
        if pd.isna(s) or s == '[]':
            return []
        # Handle np.int64(X) format
        np_nums = re2.findall(r'np\.int64\((\d+)\)', str(s))
        if np_nums:
            return [int(n) for n in np_nums]
        # Handle plain [1, 2, 3] format
        nums = re2.findall(r'\b(\d{1,2})\b', str(s))
        return [int(n) for n in nums if int(n) < 24]
    bess['charge_hours'] = bess['charge_h_ind'].apply(parse_hour_list)
    bess['discharge_hours'] = bess['discharge_h_ind'].apply(parse_hour_list)

    # For each day, check if our charge hours hit bottom-2 and discharge hit top-2
    full_days = df.groupby('date').filter(lambda x: len(x) >= 24)
    actual_ranks = full_days.groupby('date').apply(
        lambda x: pd.Series({
            'cheapest_2': sorted(x.nsmallest(2, 'da_price')['hour'].tolist()),
            'expensive_2': sorted(x.nlargest(2, 'da_price')['hour'].tolist()),
        })
    ).reset_index()

    merged = bess.merge(actual_ranks, on='date', how='left')

    charge_hits = 0
    discharge_hits = 0
    total_days = 0
    for _, row in merged.iterrows():
        cheap_val = row.get('cheapest_2')
        if cheap_val is None or (isinstance(cheap_val, float) and np.isnan(cheap_val)):
            continue
        ch = set(row['charge_hours'])
        dh = set(row['discharge_hours'])
        cheap = set(row['cheapest_2'])
        exp = set(row['expensive_2'])
        total_days += 1
        charge_hits += len(ch & cheap)
        discharge_hits += len(dh & exp)

    if total_days > 0:
        print(f"  Days analyzed: {total_days}")
        print(f"  Charge hour overlap with actual cheapest 2:  "
              f"{charge_hits}/{total_days*2} = {charge_hits/(total_days*2):.1%}")
        print(f"  Discharge hour overlap with actual expensive 2: "
              f"{discharge_hits}/{total_days*2} = {discharge_hits/(total_days*2):.1%}")

    # Distribution of charge/discharge hours
    all_charge = []
    all_discharge = []
    for _, row in bess.iterrows():
        all_charge.extend(row['charge_hours'])
        all_discharge.extend(row['discharge_hours'])

    print(f"\n  Most common charge hours:")
    ch_counts = pd.Series(all_charge).value_counts().head(5)
    for h, c in ch_counts.items():
        print(f"    Hour {h:2d}: {c:4d} days ({c/len(bess)*100:.1f}%)")

    print(f"\n  Most common discharge hours:")
    dh_counts = pd.Series(all_discharge).value_counts().head(5)
    for h, c in dh_counts.items():
        print(f"    Hour {h:2d}: {c:4d} days ({c/len(bess)*100:.1f}%)")

    # --- Chart 10: Rule breakdown bar chart ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    rules = ['R1: Weekend', 'R2: High Load Dev', 'R3: Large Persistence',
             'R4: Moderate Persistence']
    rule_data = []
    for rule in rules:
        sub = traded[traded['rule'] == rule]
        if len(sub) > 0:
            rule_data.append({
                'rule': rule.split(': ')[1],
                'accuracy': sub['correct'].mean(),
                'avg_pnl': sub['pnl'].mean(),
                'total_pnl': sub['pnl'].sum(),
                'n': len(sub),
            })
    rd = pd.DataFrame(rule_data)

    # Accuracy
    ax = axes[0]
    colors = ['#4caf50' if v > 0.55 else '#ff9800' if v > 0.50 else '#d32f2f'
              for v in rd['accuracy']]
    ax.barh(rd['rule'], rd['accuracy'], color=colors, edgecolor='white')
    ax.axvline(0.5, color='red', ls='--', alpha=0.5, label='Random')
    ax.set_xlabel('Accuracy')
    ax.set_title('Accuracy by Rule')
    for i, row in rd.iterrows():
        ax.text(row['accuracy'] + 0.01, i, f"{row['accuracy']:.1%}", va='center')

    # Avg P&L
    ax = axes[1]
    colors = ['#4caf50' if v > 0 else '#d32f2f' for v in rd['avg_pnl']]
    ax.barh(rd['rule'], rd['avg_pnl'], color=colors, edgecolor='white')
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel('Avg P&L per Trade (EUR)')
    ax.set_title('Avg P&L by Rule')

    # Total P&L
    ax = axes[2]
    colors = ['#4caf50' if v > 0 else '#d32f2f' for v in rd['total_pnl']]
    ax.barh(rd['rule'], rd['total_pnl'], color=colors, edgecolor='white')
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel('Total P&L (EUR)')
    ax.set_title('Total P&L by Rule')
    for i, row in rd.iterrows():
        ax.text(row['total_pnl'] + 200, i, f"n={row['n']}", va='center', fontsize=9)

    fig.suptitle('Signal Rule Breakdown: Which Rules Drive Performance?', fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / 'plots' / '10_rule_breakdown.png', dpi=150)
    plt.close(fig)
    print(f"\n[+] Chart saved: 10_rule_breakdown.png")

    # --- Chart 11: Rule performance by year ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for yr_idx, yr in enumerate([2025, 2026]):
        ax = axes[yr_idx]
        yr_data = []
        for rule in rules:
            sub = traded[(traded['rule'] == rule) & (traded['year'] == yr)]
            if len(sub) > 0:
                yr_data.append({
                    'rule': rule.split(': ')[1],
                    'accuracy': sub['correct'].mean(),
                    'n': len(sub),
                })
        if yr_data:
            yd = pd.DataFrame(yr_data)
            colors = ['#4caf50' if v > 0.55 else '#ff9800' if v > 0.50 else '#d32f2f'
                      for v in yd['accuracy']]
            ax.barh(yd['rule'], yd['accuracy'], color=colors, edgecolor='white')
            ax.axvline(0.5, color='red', ls='--', alpha=0.5)
            ax.set_xlabel('Accuracy')
            ax.set_title(f'{yr}')
            ax.set_xlim(0, 1)
            for i, row in yd.iterrows():
                ax.text(row['accuracy'] + 0.01, i,
                        f"{row['accuracy']:.0%} (n={row['n']})", va='center', fontsize=9)

    fig.suptitle('Rule Accuracy: 2025 vs 2026 - Is Any Rule Degrading?', fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / 'plots' / '11_rule_by_year.png', dpi=150)
    plt.close(fig)
    print(f"[+] Chart saved: 11_rule_by_year.png")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
