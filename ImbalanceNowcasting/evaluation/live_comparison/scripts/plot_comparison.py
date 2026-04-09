"""
Live model vs Ipesoft prediction comparison - March 2026.

Generates 6 plots comparing the deployed LightGBM model against
Ipesoft's native imbalance predictions and the regulation baseline.
"""

import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(PLOT_DIR, "data")
OKTE_DIR = r"C:\Users\noelp\pycharmprojects\ipesoft_eda_data\OKTE_Imbalnce"
IPESOFT_CSV = r"C:\Users\noelp\pycharmprojects\ipesoft_eda_data\data\ipesoft_predictions\ipesoft_imbalance_predictions_202603.csv"

DB_PARAMS = dict(host="127.0.0.1", port=5434, user="beam",
                 password="s%Upt%H%5vpD2gW@9r&S", dbname="beam-solar")


def load_data():
    """Load all three data sources and merge on QH + lead."""
    # -- DB predictions (model + baseline) --
    conn = psycopg2.connect(**DB_PARAMS)
    db = pd.read_sql(
        'SELECT "timestamp", prediction_mwh, lead_time_min, baseline_pred '
        'FROM beam.predictions', conn)
    conn.close()
    db['timestamp'] = pd.to_datetime(db['timestamp'], utc=True)
    db['qh_start'] = db['timestamp'].dt.floor('15min')

    # -- Ipesoft predictions --
    ipe = pd.read_csv(IPESOFT_CSV, parse_dates=['timestamp', 'qh_start'])
    ipe['qh_start'] = ipe['qh_start'].dt.tz_localize('Europe/Bratislava').dt.tz_convert('UTC')

    # -- OKTE actuals --
    actuals = pd.read_csv(
        os.path.join(OKTE_DIR, "SystemImbalance_2026-03-01_2026-03-30.csv"),
        sep=';')
    actuals = actuals.rename(columns={
        'Date': 'date', 'Settlement Term': 'period',
        'System Imbalance (MWh)': 'actual_mwh'})
    actuals['actual_mwh'] = pd.to_numeric(actuals['actual_mwh'], errors='coerce')
    actuals['date'] = pd.to_datetime(actuals['date'], format='%m/%d/%Y')
    actuals['period'] = actuals['period'].astype(int)

    dst_switch = pd.Timestamp('2026-03-29 02:00:00')
    actuals['psl'] = actuals['date'] + pd.to_timedelta(
        (actuals['period'] - 1) * 15, unit='m')
    actuals['utc_off'] = np.where(actuals['psl'] < dst_switch, 1, 2)
    actuals['qh_start'] = (
        actuals['psl'] - pd.to_timedelta(actuals['utc_off'], unit='h')
    ).dt.tz_localize('UTC')

    # -- Merge DB --
    db_m = db.merge(actuals[['qh_start', 'actual_mwh']], on='qh_start', how='inner')
    db_m = db_m[(db_m['qh_start'] >= '2026-03-01') & (db_m['qh_start'] < '2026-04-01')]
    db_m = db_m.dropna(subset=['actual_mwh', 'prediction_mwh'])

    # -- Merge Ipesoft --
    ipe_m = ipe.merge(actuals[['qh_start', 'actual_mwh']], on='qh_start', how='inner')
    ipe_m = ipe_m[(ipe_m['qh_start'] >= '2026-03-01') & (ipe_m['qh_start'] < '2026-04-01')]
    ipe_m = ipe_m.dropna(subset=['actual_mwh', 'value_mwh'])

    return db_m, ipe_m, actuals


def plot_01_mae_by_lead(db_m, ipe_m):
    """Grouped bar chart: MAE by lead time — Ipesoft, Model, and Baseline."""
    leads = [0, 3, 6, 9, 12]
    model_mae, ipe_mae, bl_mae = [], [], []

    for lead in leads:
        d = db_m[db_m['lead_time_min'] == lead]
        i = ipe_m[ipe_m['lead_time_min'] == lead]
        model_mae.append((d['prediction_mwh'] - d['actual_mwh']).abs().mean())
        bl_mae.append((d['baseline_pred'] - d['actual_mwh']).abs().mean())
        ipe_mae.append((i['value_mwh'] - i['actual_mwh']).abs().mean())

    x = np.arange(len(leads))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    # Ipesoft and Model side by side, baseline behind as reference
    ax.bar(x, bl_mae, w * 2.5, label='Baseline (regulation)', color='#e67e22',
           edgecolor='white', alpha=0.25, zorder=1)
    ax.bar(x - w/2, ipe_mae, w, label='Ipesoft', color='#7f8c8d',
           edgecolor='white', zorder=2)
    ax.bar(x + w/2, model_mae, w, label='LightGBM Model', color='#2ecc71',
           edgecolor='white', zorder=2)

    ax.set_xlabel('Lead Time (minutes)')
    ax.set_ylabel('MAE (MWh)')
    ax.set_title('Mean Absolute Error by Lead Time - March 2026')
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in leads])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for i_idx, (im, mm) in enumerate(zip(ipe_mae, model_mae)):
        ax.text(i_idx - w/2, im + 0.1, f'{im:.2f}', ha='center', va='bottom', fontsize=8)
        ax.text(i_idx + w/2, mm + 0.1, f'{mm:.2f}', ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, '01_mae_by_lead_time.png'), dpi=150)
    plt.close(fig)
    print("[+] 01_mae_by_lead_time.png")


def plot_02_error_distribution(db_m, ipe_m):
    """Overlaid histograms of errors at Lead 12 (biggest model advantage)."""
    lead = 12
    d = db_m[db_m['lead_time_min'] == lead]
    i = ipe_m[ipe_m['lead_time_min'] == lead]

    model_err = d['prediction_mwh'] - d['actual_mwh']
    ipe_err = i['value_mwh'] - i['actual_mwh']

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(-25, 25, 80)
    ax.hist(ipe_err, bins=bins, alpha=0.4, density=True, label='Ipesoft', color='#7f8c8d')
    ax.hist(model_err, bins=bins, alpha=0.6, density=True, label='LightGBM Model', color='#2ecc71')

    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Prediction Error (MWh)')
    ax.set_ylabel('Density')
    ax.set_title(f'Error Distribution at Lead {lead} min - March 2026')
    ax.legend()

    stats = (f'Model:   MAE={model_err.abs().mean():.2f}, bias={model_err.mean():+.2f}\n'
             f'Ipesoft: MAE={ipe_err.abs().mean():.2f}, bias={ipe_err.mean():+.2f}')
    ax.text(0.98, 0.95, stats, transform=ax.transAxes, ha='right', va='top',
            fontsize=9, family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, '02_error_distribution_lead12.png'), dpi=150)
    plt.close(fig)
    print("[+] 02_error_distribution_lead12.png")


def plot_03_time_series(db_m, ipe_m, actuals):
    """2-day excerpt showing actual vs predictions (Lead 12)."""
    lead = 12
    # Pick a mid-month window with interesting variation
    t0 = pd.Timestamp('2026-03-10', tz='UTC')
    t1 = pd.Timestamp('2026-03-12', tz='UTC')

    act = actuals[(actuals['qh_start'] >= t0) & (actuals['qh_start'] < t1)].sort_values('qh_start')
    d = db_m[(db_m['lead_time_min'] == lead) &
             (db_m['qh_start'] >= t0) & (db_m['qh_start'] < t1)].sort_values('qh_start')
    i = ipe_m[(ipe_m['lead_time_min'] == lead) &
              (ipe_m['qh_start'] >= t0) & (ipe_m['qh_start'] < t1)].sort_values('qh_start')

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(act['qh_start'], act['actual_mwh'], 'k-', linewidth=2, label='Actual (OKTE)', zorder=3)
    ax.plot(i['qh_start'], i['value_mwh'], '--', color='#7f8c8d', linewidth=1.2,
            label='Ipesoft', alpha=0.8)
    ax.plot(d['qh_start'], d['prediction_mwh'], '-', color='#2ecc71', linewidth=1.5,
            label='LightGBM Model', alpha=0.9)

    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.fill_between(act['qh_start'], 0, act['actual_mwh'], alpha=0.08, color='black')
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('System Imbalance (MWh)')
    ax.set_title(f'Actual vs Predictions at Lead {lead} min - March 10-12, 2026')
    ax.legend(loc='upper right')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d %H:%M'))
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, '03_time_series_lead12.png'), dpi=150)
    plt.close(fig)
    print("[+] 03_time_series_lead12.png")


def plot_04_qh_mistakes(db_m, ipe_m):
    """QH mistake distribution (0-5 wrong directions per QH)."""
    # Model
    db_m['dir_wrong'] = ((db_m['prediction_mwh'] * db_m['actual_mwh']) < 0).astype(int)
    qh_model = db_m.groupby('qh_start').agg(
        n=('dir_wrong', 'count'), wrong=('dir_wrong', 'sum')).reset_index()
    qh_model = qh_model[qh_model['n'] == 5]

    # Ipesoft
    ipe_m['dir_wrong'] = ((ipe_m['value_mwh'] * ipe_m['actual_mwh']) < 0).astype(int)
    qh_ipe = ipe_m.groupby('qh_start').agg(
        n=('dir_wrong', 'count'), wrong=('dir_wrong', 'sum')).reset_index()
    qh_ipe = qh_ipe[qh_ipe['n'] == 5]

    mistakes = list(range(6))
    model_counts = [(qh_model['wrong'] == m).sum() / len(qh_model) * 100 for m in mistakes]
    ipe_counts = [(qh_ipe['wrong'] == m).sum() / len(qh_ipe) * 100 for m in mistakes]

    x = np.arange(len(mistakes))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - w/2, ipe_counts, w, label='Ipesoft', color='#7f8c8d', edgecolor='white')
    bars2 = ax.bar(x + w/2, model_counts, w, label='LightGBM Model', color='#2ecc71', edgecolor='white')

    for bar, val in zip(bars1, ipe_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, model_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    # Mark the 3+ threshold
    ax.axvline(2.5, color='red', linestyle='--', alpha=0.5, label='Wrong threshold (3+)')

    ax.set_xlabel('Direction Mistakes per QH (out of 5 predictions)')
    ax.set_ylabel('Percentage of QH Periods')
    ax.set_title('QH Directional Mistake Distribution - March 2026')
    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in mistakes])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, '04_qh_mistake_distribution.png'), dpi=150)
    plt.close(fig)
    print("[+] 04_qh_mistake_distribution.png")


def plot_05_scatter(db_m, ipe_m):
    """Scatter: predicted vs actual for Ipesoft and Model at Lead 12, 6, 0."""
    leads = [12, 6, 0]
    fig, axes = plt.subplots(3, 2, figsize=(14, 18), sharex=True, sharey=True)

    for row, lead in enumerate(leads):
        d = db_m[db_m['lead_time_min'] == lead]
        i = ipe_m[ipe_m['lead_time_min'] == lead]

        for col, (data, pred_col, title, color) in enumerate([
            (i, 'value_mwh', 'Ipesoft', '#7f8c8d'),
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

    fig.suptitle('Predicted vs Actual - March 2026', fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, '05_scatter.png'), dpi=150)
    plt.close(fig)
    print("[+] 05_scatter.png")


def plot_06_rolling_mae(db_m, ipe_m):
    """6-hourly average MAE over the month for Lead 12."""
    lead = 12
    d = db_m[db_m['lead_time_min'] == lead].copy()
    i = ipe_m[ipe_m['lead_time_min'] == lead].copy()

    d['bin6h'] = d['qh_start'].dt.floor('6h')
    i['bin6h'] = i['qh_start'].dt.floor('6h')

    d['model_ae'] = (d['prediction_mwh'] - d['actual_mwh']).abs()
    i['ipe_ae'] = (i['value_mwh'] - i['actual_mwh']).abs()

    avg_model = d.groupby('bin6h')['model_ae'].mean()
    avg_ipe = i.groupby('bin6h')['ipe_ae'].mean()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(avg_ipe.index, avg_ipe.values, '-', color='#7f8c8d',
            linewidth=1.5, alpha=0.8, label='Ipesoft')
    ax.plot(avg_model.index, avg_model.values, '-', color='#2ecc71',
            linewidth=1.5, alpha=0.8, label='LightGBM Model')

    common_idx = avg_model.index.intersection(avg_ipe.index)
    ax.fill_between(common_idx,
                    avg_model.reindex(common_idx).values,
                    avg_ipe.reindex(common_idx).values,
                    alpha=0.15, color='#2ecc71')

    ax.set_xlabel('Date')
    ax.set_ylabel('6-Hourly MAE (MWh)')
    ax.set_title(f'6-Hourly Average MAE at Lead {lead} min - March 2026')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, '06_6hourly_mae_lead12.png'), dpi=150)
    plt.close(fig)
    print("[+] 06_6hourly_mae_lead12.png")


def save_summary_data(db_m, ipe_m):
    """Save comparison stats to CSV."""
    leads = [0, 3, 6, 9, 12]
    rows = []
    for lead in leads:
        d = db_m[db_m['lead_time_min'] == lead]
        i = ipe_m[ipe_m['lead_time_min'] == lead]
        rows.append({
            'lead_time_min': lead,
            'model_mae': (d['prediction_mwh'] - d['actual_mwh']).abs().mean(),
            'baseline_mae': (d['baseline_pred'] - d['actual_mwh']).abs().mean(),
            'ipesoft_mae': (i['value_mwh'] - i['actual_mwh']).abs().mean(),
            'model_rmse': np.sqrt(((d['prediction_mwh'] - d['actual_mwh'])**2).mean()),
            'ipesoft_rmse': np.sqrt(((i['value_mwh'] - i['actual_mwh'])**2).mean()),
            'model_corr': d['prediction_mwh'].corr(d['actual_mwh']),
            'ipesoft_corr': i['value_mwh'].corr(i['actual_mwh']),
            'model_bias': (d['prediction_mwh'] - d['actual_mwh']).mean(),
            'ipesoft_bias': (i['value_mwh'] - i['actual_mwh']).mean(),
            'model_dir_acc': ((d['prediction_mwh'] * d['actual_mwh']) > 0).mean(),
            'ipesoft_dir_acc': ((i['value_mwh'] * i['actual_mwh']) > 0).mean(),
        })
    stats = pd.DataFrame(rows)
    stats.to_csv(os.path.join(DATA_DIR, 'comparison_stats.csv'), index=False)
    print("[+] data/comparison_stats.csv")


if __name__ == '__main__':
    print("[*] Loading data...")
    db_m, ipe_m, actuals = load_data()
    print(f"[*] DB predictions: {len(db_m)}, Ipesoft: {len(ipe_m)}")

    plot_01_mae_by_lead(db_m, ipe_m)
    plot_02_error_distribution(db_m, ipe_m)
    plot_03_time_series(db_m, ipe_m, actuals)
    plot_04_qh_mistakes(db_m, ipe_m)
    plot_05_scatter(db_m, ipe_m)
    plot_06_rolling_mae(db_m, ipe_m)
    save_summary_data(db_m, ipe_m)
    print("[+] All plots saved")
