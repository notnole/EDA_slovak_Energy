"""
Multi-Lead Spread Strategy Test
================================

Tests the IDM-settlement spread strategy at multiple execution lead times:
  120min (2h, baseline), 180min (3h), 240min (4h), 300min (5h)

Test A: Same model (trained on 120min spread), different execution prices.
  -> Tests: "if we place orders earlier, do we capture more spread?"

Test B: Retrained model per lead (spread_target = settle - exec_mid_at_lead).
  -> Tests: "can we predict the spread at each lead?"

Uses the same 16-month walk-forward framework as walkforward_montecarlo.py.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from train_multi_lead import load_all_data, build_features
import train_multi_lead as tml

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
REPO_ROOT = BASE_DIR.parent
PLOT_DIR = BASE_DIR / "plots" / "eda" / "multi_lead"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

LEAD = 8  # feature lead (same for all tests)
SIZE_MW = 5.0
QH = 0.25
ENERGY = SIZE_MW * QH

LGB_PARAMS = dict(learning_rate=0.03, num_leaves=15, min_child_samples=200,
                  subsample=0.5, colsample_bytree=0.5, reg_alpha=1.0,
                  reg_lambda=10.0, n_estimators=200, verbose=-1)

# 50 selected features (from walkforward_montecarlo / feature_selection_spread)
SELECTED_50 = [
    'da_price', 'cloudcover', 'hour_cos', 'idm_vwap_lag', 'da_supply',
    'da_price_change24h', 'proxy_rmax4', 'temp_forecast_da', 'temp_national_spread',
    'temp_bratislava', 'load_rmean16', 'nowcast_momentum_h2h3', 'temp_national_change6h',
    'da_demand', 'temp_surprise_lag', 'proxy_rmean16', 'proxy_range8', 'hour_sin',
    'spread_da_imb_lag', 'prod_momentum', 'nowcast_pred_rmean4', 'nowcast_momentum_h4h5',
    'da_flow_cz', 'load_momentum', 'xborder_momentum', 'nowcast_h3', 'radiation_national',
    'da_net_import', 'proxy_rmean32', 'nowcast_trend_h2_h5', 'dow_sin', 'imb_price_rmean4',
    'reg_rmean8', 'reg_vol_rmean4', 'proxy_dev_from_hour', 'proxy_yesterday', 'prod_rmean8',
    'dow_cos', 'solar_surprise_lag', 'nowcast_h5', 'proxy_rmin4', 'nowcast_convergence',
    'reg_rmean4', 'is_weekend', 'proxy_yesterday_2', 'temp_rmean24h', 'proxy_range4',
    'proxy_lag12', 'proxy_pos_ratio_4', 'proxy_lag21', 'proxy_lag18', 'damas_fe_rmean4',
]

# Walk-forward folds (same 16 months as walkforward_montecarlo.py)
FOLDS = [
    ('2024-10-01', '2024-10-01', '2024-11-01'),
    ('2024-11-01', '2024-11-01', '2024-12-01'),
    ('2024-12-01', '2024-12-01', '2025-01-01'),
    ('2025-01-01', '2025-01-01', '2025-02-01'),
    ('2025-02-01', '2025-02-01', '2025-03-01'),
    ('2025-03-01', '2025-03-01', '2025-04-01'),
    ('2025-04-01', '2025-04-01', '2025-05-01'),
    ('2025-05-01', '2025-05-01', '2025-06-01'),
    ('2025-06-01', '2025-06-01', '2025-07-01'),
    ('2025-07-01', '2025-07-01', '2025-08-01'),
    ('2025-08-01', '2025-08-01', '2025-09-01'),
    ('2025-09-01', '2025-09-01', '2025-10-01'),
    ('2025-12-01', '2025-12-01', '2026-01-01'),
    ('2026-01-01', '2026-01-01', '2026-02-01'),
    ('2026-02-01', '2026-02-01', '2026-03-01'),
    ('2026-03-01', '2026-03-01', '2026-04-01'),
]

EXEC_LEADS = [120, 180, 240, 300]
THRESHOLD = 3


def load_ob_data():
    """Load order book data for all lead times."""
    # Original leads (65, 75, 90, 105, 120)
    ob_orig = pd.read_csv(DATA_DIR / "features" / "orderbook_qh_features.csv",
                          parse_dates=['delivery_start'])
    # Extended leads (180, 240, 300)
    ob_ext = pd.read_csv(DATA_DIR / "features" / "orderbook_qh_extended_leads.csv",
                         parse_dates=['delivery_start'])

    ob_all = {}
    for lead_min in EXEC_LEADS:
        if lead_min <= 120:
            src = ob_orig[ob_orig['lead_minutes'] == lead_min].copy()
        else:
            src = ob_ext[ob_ext['lead_minutes'] == lead_min].copy()
        src = src.set_index('delivery_start')[['bid', 'ask', 'spread', 'mid']]
        src = src[~src.index.duplicated(keep='last')]
        src.columns = [f'exec_bid_{lead_min}', f'exec_ask_{lead_min}',
                       f'exec_spread_{lead_min}', f'exec_mid_{lead_min}']
        ob_all[lead_min] = src
        print(f"  [+] Lead {lead_min}min: {len(src)} rows, "
              f"both sides: {src[f'exec_spread_{lead_min}'].notna().sum()}")

    return ob_all


def walkforward_test_a(df_base, spread_features, ob_all):
    """Test A: Same model (trained on 120min spread), different execution prices."""
    print("\n" + "=" * 70)
    print("TEST A: SAME MODEL, DIFFERENT EXECUTION LEAD")
    print("Model trained on spread_target_120 = settle - exec_mid_120")
    print("Executed at each lead's bid/ask")
    print("=" * 70)

    results = {lead: [] for lead in EXEC_LEADS}

    for train_end, test_start, test_end in FOLDS:
        # Train on 120min spread (same model for all leads)
        train = df_base[df_base.index < train_end].dropna(
            subset=['spread_target_120', f'proxy_lag{LEAD+1}'])
        train = train[train['imb_settlement_price'].abs() <= 5000]

        if len(train) < 1000:
            continue

        model = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **LGB_PARAMS)
        model.fit(train[spread_features].values, train['spread_target_120'].values)

        for exec_lead in EXEC_LEADS:
            bid_col = f'exec_bid_{exec_lead}'
            ask_col = f'exec_ask_{exec_lead}'
            spread_col = f'exec_spread_{exec_lead}'

            test = df_base[(df_base.index >= test_start) & (df_base.index < test_end)].copy()
            test = test.dropna(subset=[f'proxy_lag{LEAD+1}'])
            test = test[test[bid_col].notna() & test[ask_col].notna()]
            test = test[test[spread_col] <= 10]

            if len(test) < 10:
                continue

            test['pred'] = model.predict(test[spread_features].values)

            surplus = test['pred'] <= -THRESHOLD
            deficit = test['pred'] >= THRESHOLD
            active = test[surplus | deficit].copy()

            if len(active) < 5:
                continue

            s = surplus.reindex(active.index, fill_value=False)
            d = deficit.reindex(active.index, fill_value=False)
            active['pnl'] = 0.0
            active.loc[s, 'pnl'] = (active.loc[s, bid_col] - active.loc[s, 'imb_settlement_price']) * ENERGY
            active.loc[d, 'pnl'] = (active.loc[d, 'imb_settlement_price'] - active.loc[d, ask_col]) * ENERGY
            active['direction'] = np.where(active['pred'] > 0, 'deficit', 'surplus')

            results[exec_lead].append(active[['pnl', 'pred', 'direction']])

    return results


def walkforward_test_b(df_base, spread_features, ob_all):
    """Test B: Retrained model per lead."""
    print("\n" + "=" * 70)
    print("TEST B: RETRAINED MODEL PER EXECUTION LEAD")
    print("For each lead: spread_target = settle - exec_mid_at_lead")
    print("=" * 70)

    results = {lead: [] for lead in EXEC_LEADS}

    for train_end, test_start, test_end in FOLDS:
        for exec_lead in EXEC_LEADS:
            target_col = f'spread_target_{exec_lead}'
            bid_col = f'exec_bid_{exec_lead}'
            ask_col = f'exec_ask_{exec_lead}'
            spread_col = f'exec_spread_{exec_lead}'

            train = df_base[df_base.index < train_end].copy()
            train = train.dropna(subset=[target_col, f'proxy_lag{LEAD+1}'])
            train = train[train['imb_settlement_price'].abs() <= 5000]

            test = df_base[(df_base.index >= test_start) & (df_base.index < test_end)].copy()
            test = test.dropna(subset=[f'proxy_lag{LEAD+1}'])
            test = test[test[bid_col].notna() & test[ask_col].notna()]
            test = test[test[spread_col] <= 10]

            if len(train) < 1000 or len(test) < 10:
                continue

            model = lgb.LGBMRegressor(objective='quantile', alpha=0.50, **LGB_PARAMS)
            model.fit(train[spread_features].values, train[target_col].values)

            test['pred'] = model.predict(test[spread_features].values)

            surplus = test['pred'] <= -THRESHOLD
            deficit = test['pred'] >= THRESHOLD
            active = test[surplus | deficit].copy()

            if len(active) < 5:
                continue

            s = surplus.reindex(active.index, fill_value=False)
            d = deficit.reindex(active.index, fill_value=False)
            active['pnl'] = 0.0
            active.loc[s, 'pnl'] = (active.loc[s, bid_col] - active.loc[s, 'imb_settlement_price']) * ENERGY
            active.loc[d, 'pnl'] = (active.loc[d, 'imb_settlement_price'] - active.loc[d, ask_col]) * ENERGY
            active['direction'] = np.where(active['pred'] > 0, 'deficit', 'surplus')

            results[exec_lead].append(active[['pnl', 'pred', 'direction']])

    return results


def summarize_results(results, test_name):
    """Print summary table for a set of results."""
    print(f"\n{'='*70}")
    print(f"SUMMARY: {test_name}")
    print(f"{'='*70}")
    print(f"{'Lead':>6s} | {'EUR/day':>8s} | {'Sharpe':>6s} | {'Win%':>5s} | {'DirAcc':>6s} | "
          f"{'Trades':>6s} | {'Days':>5s} | {'Total EUR':>10s}")
    print("-" * 70)

    summary = {}
    for lead in EXEC_LEADS:
        if not results[lead]:
            print(f"{lead:>6d} | {'N/A':>8s} | {'N/A':>6s} | {'N/A':>5s} | {'N/A':>6s} | "
                  f"{'N/A':>6s} | {'N/A':>5s} | {'N/A':>10s}")
            continue

        oos = pd.concat(results[lead])
        oos_daily = oos.groupby(oos.index.date)['pnl'].sum()
        n_days = len(oos_daily)
        n_trades = len(oos)
        total = oos['pnl'].sum()
        eur_day = oos_daily.mean()
        sharpe = oos_daily.mean() / oos_daily.std() * np.sqrt(252) if oos_daily.std() > 0 else 0
        win_pct = (oos['pnl'] > 0).mean()
        # Direction accuracy: did pred sign match actual settle-mid sign?
        daily_win = (oos_daily > 0).mean()

        print(f"{lead:>6d} | {eur_day:>+8.0f} | {sharpe:>6.1f} | {win_pct:>5.0%} | "
              f"{daily_win:>6.0%} | {n_trades:>6d} | {n_days:>5d} | {total:>+10,.0f}")

        summary[lead] = {
            'oos': oos, 'daily': oos_daily, 'total': total,
            'eur_day': eur_day, 'sharpe': sharpe, 'win_pct': win_pct,
            'daily_win': daily_win, 'n_trades': n_trades, 'n_days': n_days,
        }

    return summary


def analyze_raw_opportunity(df_base):
    """Analyze raw spread magnitude at each lead time (before any prediction)."""
    print("\n" + "=" * 70)
    print("RAW OPPORTUNITY: |settlement - exec_mid| BY LEAD TIME")
    print("(What is the raw spread before any prediction?)")
    print("=" * 70)

    # Use only the OOS period (Oct 2024 - Apr 2026)
    oos = df_base[(df_base.index >= '2024-10-01') & (df_base.index < '2026-04-01')]

    print(f"\n{'Lead':>6s} | {'Mean|Spread|':>12s} | {'Median|Spread|':>14s} | {'Std':>8s} | "
          f"{'%Avail':>7s} | {'%Spread<=10':>11s} | {'Mean Spread':>11s}")
    print("-" * 85)

    for lead in EXEC_LEADS:
        mid_col = f'exec_mid_{lead}'
        spread_col = f'exec_spread_{lead}'
        target_col = f'spread_target_{lead}'

        avail = oos[mid_col].notna()
        tight = oos[spread_col].notna() & (oos[spread_col] <= 10)
        sub = oos[tight].copy()

        if len(sub) == 0:
            print(f"{lead:>6d} | {'N/A':>12s} | {'N/A':>14s} | {'N/A':>8s} | "
                  f"{avail.mean():>7.0%} | {'N/A':>11s} | {'N/A':>11s}")
            continue

        abs_spread = sub[target_col].abs()
        raw_spread = sub[target_col]

        print(f"{lead:>6d} | {abs_spread.mean():>12.1f} | {abs_spread.median():>14.1f} | "
              f"{raw_spread.std():>8.1f} | {avail.mean():>7.0%} | {tight.mean():>11.0%} | "
              f"{raw_spread.mean():>+11.1f}")


def plot_comparison(summary_a, summary_b, df_base):
    """Create comparison plots."""
    fig = plt.figure(figsize=(22, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    leads_with_data_a = [l for l in EXEC_LEADS if l in summary_a]
    leads_with_data_b = [l for l in EXEC_LEADS if l in summary_b]

    # 1. EUR/day comparison
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(EXEC_LEADS))
    w = 0.35
    vals_a = [summary_a[l]['eur_day'] if l in summary_a else 0 for l in EXEC_LEADS]
    vals_b = [summary_b[l]['eur_day'] if l in summary_b else 0 for l in EXEC_LEADS]
    bars_a = ax.bar(x - w/2, vals_a, w, label='Test A (same model)', color='steelblue', alpha=0.7)
    bars_b = ax.bar(x + w/2, vals_b, w, label='Test B (retrained)', color='coral', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{l}min\n({l/60:.0f}h)' for l in EXEC_LEADS])
    ax.set_ylabel('EUR/day')
    ax.set_title('EUR/day by Execution Lead')
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.legend(fontsize=9)

    # 2. Sharpe comparison
    ax = fig.add_subplot(gs[0, 1])
    vals_a = [summary_a[l]['sharpe'] if l in summary_a else 0 for l in EXEC_LEADS]
    vals_b = [summary_b[l]['sharpe'] if l in summary_b else 0 for l in EXEC_LEADS]
    ax.bar(x - w/2, vals_a, w, label='Test A', color='steelblue', alpha=0.7)
    ax.bar(x + w/2, vals_b, w, label='Test B', color='coral', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{l}min' for l in EXEC_LEADS])
    ax.set_ylabel('Sharpe (annualized)')
    ax.set_title('Sharpe by Execution Lead')
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.legend(fontsize=9)

    # 3. Trade count / coverage
    ax = fig.add_subplot(gs[0, 2])
    vals_a = [summary_a[l]['n_trades'] if l in summary_a else 0 for l in EXEC_LEADS]
    vals_b = [summary_b[l]['n_trades'] if l in summary_b else 0 for l in EXEC_LEADS]
    ax.bar(x - w/2, vals_a, w, label='Test A', color='steelblue', alpha=0.7)
    ax.bar(x + w/2, vals_b, w, label='Test B', color='coral', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{l}min' for l in EXEC_LEADS])
    ax.set_ylabel('Total OOS trades')
    ax.set_title('Number of Trades by Lead')
    ax.legend(fontsize=9)

    # 4-5. Equity curves for Test A and Test B
    for col_idx, (summary, label) in enumerate([(summary_a, 'Test A'), (summary_b, 'Test B')]):
        ax = fig.add_subplot(gs[1, col_idx])
        for lead in EXEC_LEADS:
            if lead not in summary:
                continue
            eq = summary[lead]['daily'].cumsum()
            ax.plot(range(len(eq)), eq.values, lw=1.5, label=f'{lead}min ({lead/60:.0f}h)')
        ax.set_xlabel('Trading Day')
        ax.set_ylabel('Cumulative P&L (EUR)')
        ax.set_title(f'{label}: Equity Curves by Lead')
        ax.legend(fontsize=9)
        ax.axhline(0, color='gray', ls='--', alpha=0.3)

    # 6. Raw opportunity: |spread| by lead
    ax = fig.add_subplot(gs[1, 2])
    oos = df_base[(df_base.index >= '2024-10-01') & (df_base.index < '2026-04-01')]
    box_data = []
    box_labels = []
    for lead in EXEC_LEADS:
        spread_col = f'exec_spread_{lead}'
        target_col = f'spread_target_{lead}'
        tight = oos[spread_col].notna() & (oos[spread_col] <= 10)
        vals = oos.loc[tight, target_col].dropna().abs()
        if len(vals) > 0:
            box_data.append(vals.clip(upper=100).values)
            box_labels.append(f'{lead}min')
    if box_data:
        bp = ax.boxplot(box_data, labels=box_labels, showfliers=False, widths=0.6)
        ax.set_ylabel('|settle - exec_mid| (EUR)')
        ax.set_title('Raw Opportunity by Lead (clipped at 100)')

    # 7-8. Monthly breakdown for best test A and test B leads
    for col_idx, (summary, label) in enumerate([(summary_a, 'Test A'), (summary_b, 'Test B')]):
        ax = fig.add_subplot(gs[2, col_idx])
        if not summary:
            continue
        best_lead = max(summary.keys(), key=lambda l: summary[l]['eur_day'])
        oos = summary[best_lead]['oos']
        monthly = oos.groupby(oos.index.to_period('M'))['pnl'].sum()
        monthly_days = oos.groupby(oos.index.to_period('M')).apply(
            lambda x: x.index.normalize().nunique())
        monthly_ppd = monthly / monthly_days
        months = monthly_ppd.index.astype(str)
        colors = ['green' if v > 0 else 'red' for v in monthly_ppd]
        ax.bar(range(len(months)), monthly_ppd.values, color=colors, alpha=0.7)
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(months, rotation=45, fontsize=7)
        ax.set_ylabel('EUR/day')
        ax.set_title(f'{label}: Monthly P&L (best lead={best_lead}min)')
        ax.axhline(0, color='gray', ls='--', alpha=0.5)

    # 9. Win rate by lead
    ax = fig.add_subplot(gs[2, 2])
    vals_a = [summary_a[l]['win_pct'] * 100 if l in summary_a else 0 for l in EXEC_LEADS]
    vals_b = [summary_b[l]['win_pct'] * 100 if l in summary_b else 0 for l in EXEC_LEADS]
    ax.bar(x - w/2, vals_a, w, label='Test A', color='steelblue', alpha=0.7)
    ax.bar(x + w/2, vals_b, w, label='Test B', color='coral', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{l}min' for l in EXEC_LEADS])
    ax.set_ylabel('Win %')
    ax.set_title('Trade Win Rate by Lead')
    ax.axhline(50, color='gray', ls='--', alpha=0.5)
    ax.legend(fontsize=9)

    fig.savefig(PLOT_DIR / "01_multi_lead_comparison.png", bbox_inches='tight', dpi=120)
    plt.close(fig)
    print(f"\n[+] Saved: {PLOT_DIR / '01_multi_lead_comparison.png'}")


def main():
    print("=" * 70)
    print("MULTI-LEAD SPREAD STRATEGY TEST")
    print("Walk-forward: 16 months OOS, threshold=%d EUR" % THRESHOLD)
    print("Execution leads: %s minutes" % EXEC_LEADS)
    print("=" * 70)

    # --- Load all data ---
    print("\n[*] Loading data...")
    data = load_all_data()
    tml.TRAIN_END = '2026-04-01'
    tml.TEST_START = '2026-04-01'
    df_base, feature_cols = build_features(data, LEAD)

    # Validate selected features
    spread_features = [f for f in SELECTED_50 if f in feature_cols]
    missing = [f for f in SELECTED_50 if f not in feature_cols]
    if missing:
        print(f"[!] Warning: {len(missing)} features not found: {missing}")
    print(f"[+] Using {len(spread_features)} features")

    # --- Load OB data for all leads ---
    print("\n[*] Loading order book data...")
    ob_all = load_ob_data()

    # Join all execution prices
    for lead_min, ob_df in ob_all.items():
        df_base = df_base.join(ob_df, how='left')

    # Settlement price
    df_base['imb_settlement_price'] = df_base['imb_settle_price']

    # Build spread targets for each lead (hourly-smoothed)
    df_base['hour_ts'] = df_base.index.floor('h')
    df_base['settle_hourly'] = df_base.groupby('hour_ts')['imb_settlement_price'].transform('mean')

    for lead_min in EXEC_LEADS:
        mid_col = f'exec_mid_{lead_min}'
        df_base[f'mid_hourly_{lead_min}'] = df_base.groupby('hour_ts')[mid_col].transform('mean')
        df_base[f'spread_target_{lead_min}'] = df_base['settle_hourly'] - df_base[f'mid_hourly_{lead_min}']

    print(f"[+] Base data: {len(df_base)} rows")

    # --- Analyze raw opportunity ---
    analyze_raw_opportunity(df_base)

    # --- Run Test A ---
    results_a = walkforward_test_a(df_base, spread_features, ob_all)
    summary_a = summarize_results(results_a, "TEST A: Same model (120min), different execution")

    # --- Run Test B ---
    results_b = walkforward_test_b(df_base, spread_features, ob_all)
    summary_b = summarize_results(results_b, "TEST B: Retrained per lead")

    # --- Monthly breakdown for each lead/test ---
    for test_name, summary in [("A", summary_a), ("B", summary_b)]:
        for lead in EXEC_LEADS:
            if lead not in summary:
                continue
            oos = summary[lead]['oos']
            monthly = oos.groupby(oos.index.to_period('M'))['pnl'].agg(['sum', 'count'])
            monthly_days = oos.groupby(oos.index.to_period('M')).apply(
                lambda x: x.index.normalize().nunique())
            monthly['days'] = monthly_days.values
            monthly['per_day'] = monthly['sum'] / monthly['days']
            print(f"\n--- Test {test_name}, Lead {lead}min: Monthly P&L ---")
            for m, r in monthly.iterrows():
                print(f"  {m}: {r['sum']:+8,.0f} EUR ({r['per_day']:+.0f}/day), "
                      f"{int(r['count'])} trades, {int(r['days'])} days")

    # --- Comparison summary ---
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(f"\n{'Test':>7s} {'Lead':>6s} | {'EUR/day':>8s} | {'Sharpe':>6s} | {'Win%':>5s} | "
          f"{'DayWin%':>7s} | {'Trades':>6s} | {'Total':>10s}")
    print("-" * 70)
    for test_name, summary in [("A", summary_a), ("B", summary_b)]:
        for lead in EXEC_LEADS:
            if lead not in summary:
                s = summary.get(lead, {})
                print(f"{'Test '+test_name:>7s} {lead:>6d} | {'N/A':>8s}")
                continue
            s = summary[lead]
            print(f"{'Test '+test_name:>7s} {lead:>6d} | {s['eur_day']:>+8.0f} | "
                  f"{s['sharpe']:>6.1f} | {s['win_pct']:>5.0%} | {s['daily_win']:>7.0%} | "
                  f"{s['n_trades']:>6d} | {s['total']:>+10,.0f}")

    # --- Plot ---
    print("\n[*] Generating plots...")
    plot_comparison(summary_a, summary_b, df_base)

    print("\n[+] Done!")


if __name__ == "__main__":
    main()
