"""
Live model evaluation — compares deployed LightGBM model against
the regulation baseline (and optionally Ipesoft predictions).

Usage:
    python plot_comparison.py                   # auto-detect latest month with OKTE data
    python plot_comparison.py --month 2026-03   # specific month
"""

import argparse
import glob as globmod
import re
import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(PLOT_DIR, "data")
OKTE_DIR = r"C:\Users\noelp\pycharmprojects\ipesoft_eda_data\OKTE_Imbalnce"
IPESOFT_DIR = r"C:\Users\noelp\pycharmprojects\ipesoft_eda_data\data\ipesoft_predictions"

DB_PARAMS = dict(host="127.0.0.1", port=5434, user="beam",
                 password="s%Upt%H%5vpD2gW@9r&S", dbname="beam-solar")


def find_okte_files(year_month):
    """Find OKTE imbalance CSVs covering a given YYYY-MM."""
    pattern = os.path.join(OKTE_DIR, f"SystemImbalance_{year_month}-*.csv")
    return sorted(globmod.glob(pattern))


def detect_latest_month():
    """Find the most recent month with OKTE actuals data."""
    files = sorted(globmod.glob(os.path.join(OKTE_DIR, "SystemImbalance_*.csv")))
    if not files:
        raise FileNotFoundError("No OKTE files found")
    # Extract start date from filename like SystemImbalance_2026-04-01_2026-04-12.csv
    m = re.search(r'SystemImbalance_(\d{4}-\d{2})-\d{2}', os.path.basename(files[-1]))
    if not m:
        raise ValueError(f"Cannot parse month from {files[-1]}")
    return m.group(1)


def load_data(year_month):
    """Load DB predictions and OKTE actuals. Optionally load Ipesoft if available."""
    ym_short = year_month.replace('-', '')  # e.g. 202604
    month_start = f"{year_month}-01"
    # Next month for filtering
    y, m = int(year_month[:4]), int(year_month[5:7])
    nm = m + 1 if m < 12 else 1
    ny = y if m < 12 else y + 1
    month_end = f"{ny}-{nm:02d}-01"

    # -- DB predictions (model + baseline) --
    conn = psycopg2.connect(**DB_PARAMS)
    db = pd.read_sql(
        'SELECT "timestamp", prediction_mwh, lead_time_min, baseline_pred '
        'FROM beam.predictions '
        f"WHERE timestamp >= '{month_start}' AND timestamp < '{month_end}'", conn)
    conn.close()
    db['timestamp'] = pd.to_datetime(db['timestamp'], utc=True)
    db['qh_start'] = db['timestamp'].dt.floor('15min')

    # -- Ipesoft predictions (optional) --
    ipesoft_csv = os.path.join(IPESOFT_DIR, f"ipesoft_imbalance_predictions_{ym_short}.csv")
    ipe = None
    if os.path.exists(ipesoft_csv):
        ipe = pd.read_csv(ipesoft_csv, parse_dates=['timestamp', 'qh_start'])
        ipe['qh_start'] = ipe['qh_start'].dt.tz_localize('Europe/Bratislava').dt.tz_convert('UTC')
        print(f"[+] Ipesoft data loaded: {ipesoft_csv}")
    else:
        print(f"[*] No Ipesoft data for {year_month}, skipping Ipesoft comparison")

    # -- OKTE actuals --
    okte_files = find_okte_files(year_month)
    if not okte_files:
        raise FileNotFoundError(f"No OKTE files found for {year_month}")
    actuals = pd.concat([pd.read_csv(f, sep=';') for f in okte_files], ignore_index=True)
    actuals = actuals.rename(columns={
        'Date': 'date', 'Settlement Term': 'period',
        'System Imbalance (MWh)': 'actual_mwh'})
    actuals['actual_mwh'] = pd.to_numeric(actuals['actual_mwh'], errors='coerce')
    actuals['date'] = pd.to_datetime(actuals['date'], format='%m/%d/%Y')
    actuals['period'] = actuals['period'].astype(int)

    # UTC offset — April is fully CEST (UTC+2)
    actuals['psl'] = actuals['date'] + pd.to_timedelta(
        (actuals['period'] - 1) * 15, unit='m')
    # Determine UTC offset per row based on Europe/Bratislava rules
    psl_local = actuals['psl'].dt.tz_localize('Europe/Bratislava', ambiguous='infer')
    actuals['utc_off'] = psl_local.apply(lambda x: x.utcoffset().total_seconds() / 3600).astype(int)
    actuals['qh_start'] = (
        actuals['psl'] - pd.to_timedelta(actuals['utc_off'], unit='h')
    ).dt.tz_localize('UTC')
    actuals = actuals.drop_duplicates(subset=['qh_start'])

    # -- Merge DB --
    db_m = db.merge(actuals[['qh_start', 'actual_mwh']], on='qh_start', how='inner')
    db_m = db_m.dropna(subset=['actual_mwh', 'prediction_mwh'])

    # -- Merge Ipesoft --
    ipe_m = None
    if ipe is not None:
        ipe_m = ipe.merge(actuals[['qh_start', 'actual_mwh']], on='qh_start', how='inner')
        ipe_m = ipe_m.dropna(subset=['actual_mwh', 'value_mwh'])

    return db_m, ipe_m, actuals, year_month


def month_label(year_month):
    """Convert '2026-04' to 'April 2026'."""
    from datetime import datetime
    dt = datetime.strptime(year_month, '%Y-%m')
    return dt.strftime('%B %Y')


def plot_01_mae_by_lead(db_m, ipe_m, year_month):
    """Grouped bar chart: MAE by lead time — Model, Baseline (and optionally Ipesoft)."""
    label = month_label(year_month)
    leads = [0, 3, 6, 9, 12]
    model_mae, bl_mae = [], []
    ipe_mae = [] if ipe_m is not None else None

    for lead in leads:
        d = db_m[db_m['lead_time_min'] == lead]
        model_mae.append((d['prediction_mwh'] - d['actual_mwh']).abs().mean())
        bl_mae.append((d['baseline_pred'] - d['actual_mwh']).abs().mean())
        if ipe_m is not None:
            i = ipe_m[ipe_m['lead_time_min'] == lead]
            ipe_mae.append((i['value_mwh'] - i['actual_mwh']).abs().mean())

    x = np.arange(len(leads))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, bl_mae, w * 2.5, label='Baseline (regulation)', color='#e67e22',
           edgecolor='white', alpha=0.25, zorder=1)
    if ipe_mae is not None:
        ax.bar(x - w/2, ipe_mae, w, label='Ipesoft', color='#7f8c8d',
               edgecolor='white', zorder=2)
    ax.bar(x + w/2, model_mae, w, label='LightGBM Model', color='#2ecc71',
           edgecolor='white', zorder=2)

    ax.set_xlabel('Lead Time (minutes)')
    ax.set_ylabel('MAE (MWh)')
    ax.set_title(f'Mean Absolute Error by Lead Time - {label}')
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in leads])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for i_idx, mm in enumerate(model_mae):
        ax.text(i_idx + w/2, mm + 0.1, f'{mm:.2f}', ha='center', va='bottom', fontsize=8)
    if ipe_mae is not None:
        for i_idx, im in enumerate(ipe_mae):
            ax.text(i_idx - w/2, im + 0.1, f'{im:.2f}', ha='center', va='bottom', fontsize=8)
    for i_idx, bm in enumerate(bl_mae):
        ax.text(i_idx, bm + 0.1, f'{bm:.2f}', ha='center', va='bottom', fontsize=8, color='#e67e22')

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, '01_mae_by_lead_time.png'), dpi=150)
    plt.close(fig)
    print("[+] 01_mae_by_lead_time.png")


def plot_02_error_distribution(db_m, ipe_m, year_month):
    """Overlaid histograms of errors at Lead 12 (biggest model advantage)."""
    label = month_label(year_month)
    lead = 12
    d = db_m[db_m['lead_time_min'] == lead]
    model_err = d['prediction_mwh'] - d['actual_mwh']
    bl_err = d['baseline_pred'] - d['actual_mwh']

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(-25, 25, 80)
    ax.hist(bl_err, bins=bins, alpha=0.3, density=True, label='Baseline', color='#e67e22')
    if ipe_m is not None:
        i = ipe_m[ipe_m['lead_time_min'] == lead]
        ipe_err = i['value_mwh'] - i['actual_mwh']
        ax.hist(ipe_err, bins=bins, alpha=0.4, density=True, label='Ipesoft', color='#7f8c8d')
    ax.hist(model_err, bins=bins, alpha=0.6, density=True, label='LightGBM Model', color='#2ecc71')

    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Prediction Error (MWh)')
    ax.set_ylabel('Density')
    ax.set_title(f'Error Distribution at Lead {lead} min - {label}')
    ax.legend()

    stats = f'Model:    MAE={model_err.abs().mean():.2f}, bias={model_err.mean():+.2f}\nBaseline: MAE={bl_err.abs().mean():.2f}, bias={bl_err.mean():+.2f}'
    if ipe_m is not None:
        stats += f'\nIpesoft:  MAE={ipe_err.abs().mean():.2f}, bias={ipe_err.mean():+.2f}'
    ax.text(0.98, 0.95, stats, transform=ax.transAxes, ha='right', va='top',
            fontsize=9, family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, '02_error_distribution_lead12.png'), dpi=150)
    plt.close(fig)
    print("[+] 02_error_distribution_lead12.png")


def plot_03_time_series(db_m, ipe_m, actuals, year_month):
    """2-day excerpt showing actual vs predictions (Lead 12)."""
    label = month_label(year_month)
    lead = 12
    # Pick a 2-day window from the middle of available data
    dates = actuals['qh_start'].dt.date.unique()
    mid = len(dates) // 2
    t0 = pd.Timestamp(str(dates[mid]), tz='UTC')
    t1 = t0 + pd.Timedelta(days=2)

    act = actuals[(actuals['qh_start'] >= t0) & (actuals['qh_start'] < t1)].sort_values('qh_start')
    d = db_m[(db_m['lead_time_min'] == lead) &
             (db_m['qh_start'] >= t0) & (db_m['qh_start'] < t1)].sort_values('qh_start')

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(act['qh_start'], act['actual_mwh'], 'k-', linewidth=2, label='Actual (OKTE)', zorder=3)
    if ipe_m is not None:
        i = ipe_m[(ipe_m['lead_time_min'] == lead) &
                  (ipe_m['qh_start'] >= t0) & (ipe_m['qh_start'] < t1)].sort_values('qh_start')
        ax.plot(i['qh_start'], i['value_mwh'], '--', color='#7f8c8d', linewidth=1.2,
                label='Ipesoft', alpha=0.8)
    ax.plot(d['qh_start'], d['baseline_pred'], '--', color='#e67e22', linewidth=1.2,
            label='Baseline', alpha=0.7)
    ax.plot(d['qh_start'], d['prediction_mwh'], '-', color='#2ecc71', linewidth=1.5,
            label='LightGBM Model', alpha=0.9)

    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.fill_between(act['qh_start'], 0, act['actual_mwh'], alpha=0.08, color='black')
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('System Imbalance (MWh)')
    t0_str = t0.strftime('%b %d')
    t1_str = t1.strftime('%b %d')
    ax.set_title(f'Actual vs Predictions at Lead {lead} min - {t0_str}-{t1_str}, {year_month[:4]}')
    ax.legend(loc='upper right')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d %H:%M'))
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, '03_time_series_lead12.png'), dpi=150)
    plt.close(fig)
    print("[+] 03_time_series_lead12.png")


def plot_04_qh_mistakes(db_m, ipe_m, year_month):
    """QH mistake distribution (0-5 wrong directions per QH)."""
    label = month_label(year_month)
    # Model
    db_m = db_m.copy()
    db_m['dir_wrong'] = ((db_m['prediction_mwh'] * db_m['actual_mwh']) < 0).astype(int)
    qh_model = db_m.groupby('qh_start').agg(
        n=('dir_wrong', 'count'), wrong=('dir_wrong', 'sum')).reset_index()
    qh_model = qh_model[qh_model['n'] == 5]

    # Baseline
    db_m['bl_wrong'] = ((db_m['baseline_pred'] * db_m['actual_mwh']) < 0).astype(int)
    qh_bl = db_m.groupby('qh_start').agg(
        n=('bl_wrong', 'count'), wrong=('bl_wrong', 'sum')).reset_index()
    qh_bl = qh_bl[qh_bl['n'] == 5]

    mistakes = list(range(6))
    model_counts = [(qh_model['wrong'] == m).sum() / len(qh_model) * 100 for m in mistakes]
    bl_counts = [(qh_bl['wrong'] == m).sum() / len(qh_bl) * 100 for m in mistakes]

    x = np.arange(len(mistakes))
    has_ipe = ipe_m is not None
    if has_ipe:
        ipe_m = ipe_m.copy()
        ipe_m['dir_wrong'] = ((ipe_m['value_mwh'] * ipe_m['actual_mwh']) < 0).astype(int)
        qh_ipe = ipe_m.groupby('qh_start').agg(
            n=('dir_wrong', 'count'), wrong=('dir_wrong', 'sum')).reset_index()
        qh_ipe = qh_ipe[qh_ipe['n'] == 5]
        ipe_counts = [(qh_ipe['wrong'] == m).sum() / len(qh_ipe) * 100 for m in mistakes]

    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w, bl_counts, w, label='Baseline', color='#e67e22', edgecolor='white', alpha=0.6)
    if has_ipe:
        ax.bar(x, ipe_counts, w, label='Ipesoft', color='#7f8c8d', edgecolor='white')
    ax.bar(x + (0 if not has_ipe else w), model_counts, w,
           label='LightGBM Model', color='#2ecc71', edgecolor='white')

    for bar_x, val in zip(x + (0 if not has_ipe else w), model_counts):
        ax.text(bar_x, val + 0.5, f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.axvline(2.5, color='red', linestyle='--', alpha=0.5, label='Wrong threshold (3+)')

    ax.set_xlabel('Direction Mistakes per QH (out of 5 predictions)')
    ax.set_ylabel('Percentage of QH Periods')
    ax.set_title(f'QH Directional Mistake Distribution - {label}')
    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in mistakes])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, '04_qh_mistake_distribution.png'), dpi=150)
    plt.close(fig)
    print("[+] 04_qh_mistake_distribution.png")


def plot_05_scatter(db_m, ipe_m, year_month):
    """Scatter: predicted vs actual for Baseline and Model at Lead 12, 6, 0."""
    label = month_label(year_month)
    leads = [12, 6, 0]
    fig, axes = plt.subplots(3, 2, figsize=(14, 18), sharex=True, sharey=True)

    for row, lead in enumerate(leads):
        d = db_m[db_m['lead_time_min'] == lead]

        for col, (data, pred_col, title, color) in enumerate([
            (d, 'baseline_pred', 'Baseline (regulation)', '#e67e22'),
            (d, 'prediction_mwh', 'LightGBM Model', '#2ecc71'),
        ]):
            ax = axes[row, col]
            ax.scatter(data['actual_mwh'], data[pred_col], alpha=0.15, s=8, color=color)
            lims = [-50, 50]
            ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect('equal')
            ax.grid(alpha=0.3)

            if row == 2:
                ax.set_xlabel('Actual Imbalance (MWh)')
            if col == 0:
                ax.set_ylabel(f'Lead {lead} min\nPredicted (MWh)')
            if row == 0:
                ax.set_title(title, fontsize=13)

            corr = data[pred_col].corr(data['actual_mwh'])
            mae = (data[pred_col] - data['actual_mwh']).abs().mean()
            ax.text(0.05, 0.95, f'r = {corr:.3f}\nMAE = {mae:.2f}',
                    transform=ax.transAxes, va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle(f'Predicted vs Actual - {label}', fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, '05_scatter.png'), dpi=150)
    plt.close(fig)
    print("[+] 05_scatter.png")


def plot_06_rolling_mae(db_m, ipe_m, year_month):
    """6-hourly average MAE over the month for Lead 12."""
    label = month_label(year_month)
    lead = 12
    d = db_m[db_m['lead_time_min'] == lead].copy()

    d['bin6h'] = d['qh_start'].dt.floor('6h')
    d['model_ae'] = (d['prediction_mwh'] - d['actual_mwh']).abs()
    d['bl_ae'] = (d['baseline_pred'] - d['actual_mwh']).abs()

    avg_model = d.groupby('bin6h')['model_ae'].mean()
    avg_bl = d.groupby('bin6h')['bl_ae'].mean()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(avg_bl.index, avg_bl.values, '-', color='#e67e22',
            linewidth=1.5, alpha=0.6, label='Baseline')
    if ipe_m is not None:
        i = ipe_m[ipe_m['lead_time_min'] == lead].copy()
        i['bin6h'] = i['qh_start'].dt.floor('6h')
        i['ipe_ae'] = (i['value_mwh'] - i['actual_mwh']).abs()
        avg_ipe = i.groupby('bin6h')['ipe_ae'].mean()
        ax.plot(avg_ipe.index, avg_ipe.values, '-', color='#7f8c8d',
                linewidth=1.5, alpha=0.8, label='Ipesoft')
    ax.plot(avg_model.index, avg_model.values, '-', color='#2ecc71',
            linewidth=1.5, alpha=0.8, label='LightGBM Model')

    ax.fill_between(avg_model.index, avg_model.values, avg_bl.values,
                    alpha=0.15, color='#2ecc71')

    ax.set_xlabel('Date')
    ax.set_ylabel('6-Hourly MAE (MWh)')
    ax.set_title(f'6-Hourly Average MAE at Lead {lead} min - {label}')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, '06_6hourly_mae_lead12.png'), dpi=150)
    plt.close(fig)
    print("[+] 06_6hourly_mae_lead12.png")


def save_summary_data(db_m, ipe_m, year_month):
    """Save comparison stats to CSV."""
    leads = [0, 3, 6, 9, 12]
    rows = []
    for lead in leads:
        d = db_m[db_m['lead_time_min'] == lead]
        row = {
            'lead_time_min': lead,
            'month': year_month,
            'model_mae': (d['prediction_mwh'] - d['actual_mwh']).abs().mean(),
            'baseline_mae': (d['baseline_pred'] - d['actual_mwh']).abs().mean(),
            'model_rmse': np.sqrt(((d['prediction_mwh'] - d['actual_mwh'])**2).mean()),
            'baseline_rmse': np.sqrt(((d['baseline_pred'] - d['actual_mwh'])**2).mean()),
            'model_corr': d['prediction_mwh'].corr(d['actual_mwh']),
            'baseline_corr': d['baseline_pred'].corr(d['actual_mwh']),
            'model_bias': (d['prediction_mwh'] - d['actual_mwh']).mean(),
            'baseline_bias': (d['baseline_pred'] - d['actual_mwh']).mean(),
            'model_dir_acc': ((d['prediction_mwh'] * d['actual_mwh']) > 0).mean(),
            'baseline_dir_acc': ((d['baseline_pred'] * d['actual_mwh']) > 0).mean(),
            'n_obs': len(d),
        }
        if ipe_m is not None:
            i = ipe_m[ipe_m['lead_time_min'] == lead]
            row.update({
                'ipesoft_mae': (i['value_mwh'] - i['actual_mwh']).abs().mean(),
                'ipesoft_rmse': np.sqrt(((i['value_mwh'] - i['actual_mwh'])**2).mean()),
                'ipesoft_corr': i['value_mwh'].corr(i['actual_mwh']),
                'ipesoft_bias': (i['value_mwh'] - i['actual_mwh']).mean(),
                'ipesoft_dir_acc': ((i['value_mwh'] * i['actual_mwh']) > 0).mean(),
            })
        rows.append(row)
    stats = pd.DataFrame(rows)
    stats.to_csv(os.path.join(DATA_DIR, 'comparison_stats.csv'), index=False)
    print("[+] data/comparison_stats.csv")

    # Print summary table
    print(f"\n{'='*65}")
    print(f"  Live Model Evaluation - {month_label(year_month)}")
    print(f"{'='*65}")
    print(f"  {'Lead':>4s}  {'Model MAE':>10s}  {'Baseline MAE':>12s}  {'Improvement':>11s}  {'N':>5s}")
    print(f"  {'----':>4s}  {'---------':>10s}  {'------------':>12s}  {'-----------':>11s}  {'---':>5s}")
    for _, r in stats.iterrows():
        imp = (1 - r['model_mae'] / r['baseline_mae']) * 100
        print(f"  {int(r['lead_time_min']):>4d}  {r['model_mae']:>10.2f}  {r['baseline_mae']:>12.2f}  {imp:>+10.1f}%  {int(r['n_obs']):>5d}")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Live model evaluation')
    parser.add_argument('--month', type=str, default=None,
                        help='Year-month to evaluate (e.g. 2026-04). Auto-detects if omitted.')
    args = parser.parse_args()

    year_month = args.month or detect_latest_month()
    print(f"[*] Evaluating: {month_label(year_month)}")
    print("[*] Loading data...")
    db_m, ipe_m, actuals, year_month = load_data(year_month)
    ipe_count = len(ipe_m) if ipe_m is not None else 0
    print(f"[*] DB predictions: {len(db_m)}, Ipesoft: {ipe_count}, Actuals QHs: {len(actuals)}")

    plot_01_mae_by_lead(db_m, ipe_m, year_month)
    plot_02_error_distribution(db_m, ipe_m, year_month)
    plot_03_time_series(db_m, ipe_m, actuals, year_month)
    plot_04_qh_mistakes(db_m, ipe_m, year_month)
    plot_05_scatter(db_m, ipe_m, year_month)
    plot_06_rolling_mae(db_m, ipe_m, year_month)
    save_summary_data(db_m, ipe_m, year_month)
    print("[+] All plots saved")
