"""
Weekly retrain simulation — would retraining have helped?

Compares:
- Static model (trained on data up to 2026-01-22, never retrained)
- Weekly retrained model (retrained each Monday on all data through previous Sunday)

Uses the exact same feature engineering and hyperparameters as train_v4_best_model.ipynb.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import psycopg2
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE = Path(r"C:\Users\noelp\pycharmprojects\ipesoft_eda_data")
FEATURES_DIR = BASE / "data" / "features"
MASTER_DIR = BASE / "data" / "master"
PLOT_DIR = BASE / "ImbalanceNowcasting" / "evaluation" / "live_comparison"

DB_PARAMS = dict(host="127.0.0.1", port=5434, user="beam",
                 password="s%Upt%H%5vpD2gW@9r&S", dbname="beam-solar")

# Same hyperparameters as v4 notebook
PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'max_depth': 6,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 50,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'seed': 42,
}

LEAD_TIMES = [0, 3, 6, 9, 12]
ORIGINAL_TRAIN_CUTOFF = pd.Timestamp('2026-01-22')


# ---- Feature engineering (copied from v4 notebook) ----

def get_features_for_lead(lead_time):
    core = ['baseline_pred', 'reg_cumulative_mean']
    within_period = ['reg_std', 'reg_range', 'reg_trend']
    hist_reg_basic = ['reg_hist_mean_10', 'reg_hist_std_10', 'reg_hist_trend_10']
    hist_reg_extended = ['reg_hist_min_10', 'reg_hist_max_10', 'reg_hist_range_10',
                         'reg_hist_mean_20', 'reg_hist_std_20']
    hist_reg_momentum = ['reg_momentum', 'reg_acceleration']
    proxy_basic = ['proxy_lag1', 'proxy_rolling_mean4']
    proxy_extended = ['proxy_lag2', 'proxy_lag3', 'proxy_rolling_std4',
                      'proxy_rolling_mean10', 'proxy_rolling_std10']
    proxy_sign = ['proxy_last_sign', 'proxy_last_positive', 'proxy_consecutive_same_sign',
                  'proxy_prop_positive_4', 'proxy_prop_positive_10']
    proxy_momentum = ['proxy_momentum', 'proxy_acceleration', 'proxy_deviation_from_mean']
    proxy_volatility = ['proxy_volatility_ratio', 'proxy_high_volatility']
    time_basic = ['hour_sin', 'hour_cos']
    time_extended = ['is_weekend', 'dow_sin', 'dow_cos']
    other = ['load_deviation']

    if lead_time == 12:
        return (core + hist_reg_basic + hist_reg_extended + hist_reg_momentum +
                proxy_basic + proxy_extended + proxy_sign + proxy_momentum +
                proxy_volatility + time_basic + time_extended + other)
    elif lead_time == 9:
        return (core + within_period + hist_reg_basic + hist_reg_extended +
                proxy_basic + proxy_extended + proxy_sign[:3] + proxy_momentum[:2] +
                time_basic + other)
    elif lead_time == 6:
        return (core + within_period + hist_reg_basic[:2] +
                proxy_basic + proxy_extended[:2] + proxy_sign[:2] + time_basic)
    elif lead_time == 3:
        return (core + within_period + hist_reg_basic[:1] +
                proxy_basic + proxy_sign[:1] + time_basic[:1])
    else:
        return (core + within_period + proxy_basic[:1] + proxy_sign[:1])


def build_feature_df(reg_df, load_df, label_df):
    """Full feature engineering pipeline matching the v4 notebook exactly."""
    # Historical regulation features
    reg_df = reg_df.sort_values('datetime').copy()
    reg_df['reg_hist_mean_10'] = reg_df['regulation_mw'].shift(1).rolling(10).mean()
    reg_df['reg_hist_std_10'] = reg_df['regulation_mw'].shift(1).rolling(10).std()
    reg_df['reg_hist_min_10'] = reg_df['regulation_mw'].shift(1).rolling(10).min()
    reg_df['reg_hist_max_10'] = reg_df['regulation_mw'].shift(1).rolling(10).max()
    reg_df['reg_hist_range_10'] = reg_df['reg_hist_max_10'] - reg_df['reg_hist_min_10']
    reg_df['reg_hist_trend_10'] = reg_df['regulation_mw'].shift(1) - reg_df['regulation_mw'].shift(10)
    reg_df['reg_hist_mean_20'] = reg_df['regulation_mw'].shift(1).rolling(20).mean()
    reg_df['reg_hist_std_20'] = reg_df['regulation_mw'].shift(1).rolling(20).std()
    reg_df['reg_momentum'] = reg_df['regulation_mw'].shift(1) - reg_df['regulation_mw'].shift(2)
    reg_df['reg_acceleration'] = reg_df['reg_momentum'] - reg_df['reg_momentum'].shift(1)

    # Align to settlement periods
    reg_df['datetime_floor'] = reg_df['datetime'].dt.floor('3min')
    reg_df['settlement_end'] = reg_df['datetime_floor'].dt.ceil('15min')
    mask = reg_df['datetime_floor'] == reg_df['settlement_end']
    reg_df.loc[mask, 'settlement_end'] = reg_df.loc[mask, 'datetime_floor'] + pd.Timedelta(minutes=15)
    reg_df['settlement_start'] = reg_df['settlement_end'] - pd.Timedelta(minutes=15)
    reg_df['minute_in_qh'] = (reg_df['datetime_floor'] - reg_df['settlement_start']).dt.total_seconds() / 60

    pivot_reg = reg_df.pivot_table(
        index='settlement_start', columns='minute_in_qh',
        values='regulation_mw', aggfunc='first'
    ).reset_index()
    pivot_reg.columns = ['datetime'] + [f'reg_min{int(c)}' for c in pivot_reg.columns[1:]]

    hist_cols = ['reg_hist_mean_10', 'reg_hist_std_10', 'reg_hist_min_10', 'reg_hist_max_10',
                 'reg_hist_range_10', 'reg_hist_trend_10', 'reg_hist_mean_20', 'reg_hist_std_20',
                 'reg_momentum', 'reg_acceleration']
    reg_min0 = reg_df[reg_df['minute_in_qh'] == 0][['settlement_start'] + hist_cols].copy()
    reg_min0 = reg_min0.rename(columns={'settlement_start': 'datetime'})

    df = pd.merge(label_df, pivot_reg, on='datetime', how='inner')
    df = pd.merge(df, reg_min0, on='datetime', how='left')

    # Load alignment
    load_df = load_df.copy()
    load_df['datetime_floor'] = load_df['datetime'].dt.floor('3min')
    load_df['settlement_end'] = load_df['datetime_floor'].dt.ceil('15min')
    mask = load_df['datetime_floor'] == load_df['settlement_end']
    load_df.loc[mask, 'settlement_end'] = load_df.loc[mask, 'datetime_floor'] + pd.Timedelta(minutes=15)
    load_df['settlement_start'] = load_df['settlement_end'] - pd.Timedelta(minutes=15)
    load_df['minute_in_qh'] = (load_df['datetime_floor'] - load_df['settlement_start']).dt.total_seconds() / 60
    load_pivot = load_df.pivot_table(
        index='settlement_start', columns='minute_in_qh',
        values='load_mw', aggfunc='first'
    ).reset_index()
    load_pivot.columns = ['datetime'] + [f'load_min{int(c)}' for c in load_pivot.columns[1:]]
    df = pd.merge(df, load_pivot, on='datetime', how='left')

    # Time features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Proxy lag features
    df = df.sort_values('datetime').copy()
    reg_cols = ['reg_min0', 'reg_min3', 'reg_min6', 'reg_min9', 'reg_min12']
    available_cols = [c for c in reg_cols if c in df.columns]
    df['period_proxy'] = -0.25 * df[available_cols].mean(axis=1)

    df['proxy_lag1'] = df['period_proxy'].shift(1)
    df['proxy_lag2'] = df['period_proxy'].shift(2)
    df['proxy_lag3'] = df['period_proxy'].shift(3)
    df['proxy_lag4'] = df['period_proxy'].shift(4)
    df['proxy_rolling_mean4'] = df['period_proxy'].shift(1).rolling(4).mean()
    df['proxy_rolling_std4'] = df['period_proxy'].shift(1).rolling(4).std()
    df['proxy_rolling_mean10'] = df['period_proxy'].shift(1).rolling(10).mean()
    df['proxy_rolling_std10'] = df['period_proxy'].shift(1).rolling(10).std()
    df['proxy_rolling_min10'] = df['period_proxy'].shift(1).rolling(10).min()
    df['proxy_rolling_max10'] = df['period_proxy'].shift(1).rolling(10).max()
    df['proxy_last_sign'] = np.sign(df['period_proxy'].shift(1))
    df['proxy_last_positive'] = (df['period_proxy'].shift(1) > 0).astype(int)
    df['proxy_last_negative'] = (df['period_proxy'].shift(1) < 0).astype(int)

    signs = np.sign(df['period_proxy'])
    sign_change = (signs != signs.shift(1)).astype(int)
    sign_change.iloc[0] = 1
    groups = sign_change.cumsum()
    df['proxy_consecutive_same_sign'] = groups.groupby(groups).cumcount().shift(1)

    positive = (df['period_proxy'] > 0).astype(float)
    df['proxy_prop_positive_4'] = positive.shift(1).rolling(4).mean()
    df['proxy_prop_positive_10'] = positive.shift(1).rolling(10).mean()
    df['proxy_momentum'] = df['period_proxy'].shift(1) - df['period_proxy'].shift(2)
    df['proxy_acceleration'] = df['proxy_momentum'] - df['proxy_momentum'].shift(1)
    df['proxy_deviation_from_mean'] = df['period_proxy'].shift(1) - df['proxy_rolling_mean10']

    short_std = df['period_proxy'].shift(1).rolling(4).std()
    long_std = df['period_proxy'].shift(1).rolling(20, min_periods=5).std()
    df['proxy_volatility_ratio'] = short_std / long_std.clip(lower=0.1)
    df['proxy_high_volatility'] = (df['proxy_volatility_ratio'] > 1.5).astype(int)

    return df


def compute_lead_features(df, lead_time, load_expected):
    """Compute lead-time specific features."""
    result = df.copy()
    available_minutes = {12: [0], 9: [0, 3], 6: [0, 3, 6], 3: [0, 3, 6, 9], 0: [0, 3, 6, 9, 12]}
    mins = available_minutes[lead_time]
    reg_cols = [f'reg_min{m}' for m in mins]
    load_cols = [f'load_min{m}' for m in mins if f'load_min{m}' in df.columns]

    result['reg_cumulative_mean'] = df[reg_cols].mean(axis=1)

    if lead_time == 12:
        result['baseline_pred'] = -0.25 * df['reg_min0']
    elif lead_time == 9:
        result['baseline_pred'] = -0.25 * (0.8 * df['reg_min3'] + 0.2 * df['reg_min0'])
    elif lead_time == 6:
        result['baseline_pred'] = -0.25 * (0.6 * df['reg_min6'] + 0.2 * df['reg_min3'] + 0.2 * df['reg_min0'])
    elif lead_time == 3:
        result['baseline_pred'] = -0.25 * (0.4 * df['reg_min9'] + 0.2 * df['reg_min6'] +
                                            0.2 * df['reg_min3'] + 0.2 * df['reg_min0'])
    else:
        result['baseline_pred'] = -0.25 * df[reg_cols].mean(axis=1)

    if len(load_cols) > 0:
        load_mean = df[load_cols].mean(axis=1)
        temp = df[['hour', 'is_weekend']].copy()
        temp['minute'] = mins[0]
        temp = temp.merge(load_expected.reset_index(), on=['hour', 'minute', 'is_weekend'], how='left')
        result['load_deviation'] = load_mean - temp['expected_load'].values
    else:
        result['load_deviation'] = 0

    if len(reg_cols) >= 2:
        result['reg_std'] = df[reg_cols].std(axis=1)
        result['reg_range'] = df[reg_cols].max(axis=1) - df[reg_cols].min(axis=1)
        result['reg_trend'] = df[reg_cols[-1]] - df[reg_cols[0]]
        result['reg_min'] = df[reg_cols].min(axis=1)
        result['reg_max'] = df[reg_cols].max(axis=1)
    else:
        result['reg_std'] = 0
        result['reg_range'] = 0
        result['reg_trend'] = 0
        result['reg_min'] = df[reg_cols[0]]
        result['reg_max'] = df[reg_cols[0]]

    return result


def train_models(df, train_cutoff, load_expected):
    """Train 5 models (one per lead time) on data before cutoff."""
    models = {}
    for lead in LEAD_TIMES:
        lead_df = compute_lead_features(df, lead, load_expected)
        train = lead_df[lead_df['datetime'] < train_cutoff].copy()
        feature_cols = get_features_for_lead(lead)
        feature_cols = [c for c in feature_cols if c in lead_df.columns]

        clean = train.dropna(subset=feature_cols + ['imbalance'])
        if len(clean) < 100:
            print(f"  [!] Lead {lead}: only {len(clean)} training rows, skipping")
            continue

        X = clean[feature_cols].values
        y = clean['imbalance'].values
        ds = lgb.Dataset(X, label=y)
        model = lgb.train(PARAMS, ds, num_boost_round=500)
        models[lead] = (model, feature_cols)

    return models


def evaluate_models(models, df, eval_start, eval_end, load_expected):
    """Evaluate models on a date range, return per-lead MAE."""
    results = {}
    for lead in LEAD_TIMES:
        if lead not in models:
            continue
        model, feature_cols = models[lead]
        lead_df = compute_lead_features(df, lead, load_expected)
        test = lead_df[(lead_df['datetime'] >= eval_start) & (lead_df['datetime'] < eval_end)].copy()
        clean = test.dropna(subset=feature_cols + ['imbalance'])
        if len(clean) == 0:
            continue

        X = clean[feature_cols].values
        y_pred = model.predict(X)
        y_true = clean['imbalance'].values
        baseline = clean['baseline_pred'].values

        results[lead] = {
            'mae': np.mean(np.abs(y_pred - y_true)),
            'baseline_mae': np.mean(np.abs(baseline - y_true)),
            'n': len(clean),
        }
    return results


def main():
    print("[*] Loading data...")
    reg_df = pd.read_csv(FEATURES_DIR / 'regulation_3min.csv', parse_dates=['datetime'])
    load_df = pd.read_csv(FEATURES_DIR / 'load_3min.csv', parse_dates=['datetime'])
    label_df = pd.read_csv(MASTER_DIR / 'master_imbalance_data.csv', parse_dates=['datetime'])
    label_df = label_df[['datetime', 'System Imbalance (MWh)']].rename(
        columns={'System Imbalance (MWh)': 'imbalance'})

    print(f"[*] Reg: {len(reg_df):,}, Load: {len(load_df):,}, Labels: {len(label_df):,}")

    # Compute expected load from 2024 training data
    load_tmp = load_df.copy()
    load_tmp['hour'] = load_tmp['datetime'].dt.hour
    load_tmp['minute'] = load_tmp['datetime'].dt.minute
    load_tmp['is_weekend'] = load_tmp['datetime'].dt.dayofweek >= 5
    load_expected = load_tmp[load_tmp['datetime'].dt.year == 2024].groupby(
        ['hour', 'minute', 'is_weekend'])['load_mw'].mean()
    load_expected.name = 'expected_load'

    print("[*] Building feature dataframe...")
    df = build_feature_df(reg_df, load_df, label_df)
    print(f"[*] Feature df: {len(df):,} rows, {len(df.columns)} columns")

    # Define evaluation weeks (Mondays)
    # Live model started ~Feb 28, OKTE actuals available through Apr 12
    eval_start = pd.Timestamp('2026-03-02')  # First Monday in March
    eval_end = pd.Timestamp('2026-04-13')    # Through Apr 12

    weeks = pd.date_range(eval_start, eval_end, freq='W-MON')
    if weeks[-1] < eval_end:
        weeks = weeks.append(pd.DatetimeIndex([eval_end]))

    print(f"\n[*] Simulation period: {eval_start.date()} to {eval_end.date()}")
    print(f"[*] Evaluation weeks: {len(weeks) - 1}")

    # ---- Train static model (original cutoff) ----
    print(f"\n[*] Training STATIC model (cutoff: {ORIGINAL_TRAIN_CUTOFF.date()})...")
    static_models = train_models(df, ORIGINAL_TRAIN_CUTOFF, load_expected)
    print(f"[+] Static models trained: {list(static_models.keys())}")

    # ---- Simulate weekly retraining ----
    all_results = []

    for i in range(len(weeks) - 1):
        week_start = weeks[i]
        week_end = weeks[i + 1]

        # Retrained model: train on everything up to this week
        retrain_cutoff = week_start
        print(f"\n{'='*60}")
        print(f"Week {i+1}: {week_start.date()} to {week_end.date()}")
        print(f"  Retrain cutoff: {retrain_cutoff.date()}")

        retrained_models = train_models(df, retrain_cutoff, load_expected)

        # Evaluate both on this week
        static_res = evaluate_models(static_models, df, week_start, week_end, load_expected)
        retrained_res = evaluate_models(retrained_models, df, week_start, week_end, load_expected)

        for lead in LEAD_TIMES:
            if lead in static_res and lead in retrained_res:
                sr = static_res[lead]
                rr = retrained_res[lead]
                imp = (1 - rr['mae'] / sr['mae']) * 100
                all_results.append({
                    'week_start': week_start,
                    'week_end': week_end,
                    'lead': lead,
                    'static_mae': sr['mae'],
                    'retrained_mae': rr['mae'],
                    'baseline_mae': sr['baseline_mae'],
                    'improvement_pct': imp,
                    'n': sr['n'],
                })
                if lead == 12:
                    print(f"  Lead 12: static={sr['mae']:.2f}, retrained={rr['mae']:.2f}, "
                          f"improvement={imp:+.1f}%, baseline={sr['baseline_mae']:.2f}")

    # ---- Summary ----
    res_df = pd.DataFrame(all_results)
    res_df.to_csv(PLOT_DIR / 'data' / 'weekly_retrain_sim.csv', index=False)

    print(f"\n{'='*60}")
    print("WEEKLY RETRAIN SIMULATION SUMMARY")
    print(f"{'='*60}")

    for lead in LEAD_TIMES:
        sub = res_df[res_df['lead'] == lead]
        if len(sub) == 0:
            continue
        # Weighted average by n
        total_n = sub['n'].sum()
        static_avg = (sub['static_mae'] * sub['n']).sum() / total_n
        retrained_avg = (sub['retrained_mae'] * sub['n']).sum() / total_n
        baseline_avg = (sub['baseline_mae'] * sub['n']).sum() / total_n
        imp = (1 - retrained_avg / static_avg) * 100
        print(f"  Lead {lead:2d}: static={static_avg:.3f}, retrained={retrained_avg:.3f}, "
              f"baseline={baseline_avg:.3f}, retrain_improvement={imp:+.1f}%")

    print(f"\nWeeks where retraining helped at Lead 12:")
    lead12 = res_df[res_df['lead'] == 12]
    for _, row in lead12.iterrows():
        marker = "<<" if row['improvement_pct'] > 0 else ""
        print(f"  {row['week_start'].date()}: static={row['static_mae']:.2f}, "
              f"retrained={row['retrained_mae']:.2f}, {row['improvement_pct']:+.1f}% {marker}")

    # ---- Plot ----
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for lead, ax in zip([12, 0], axes):
        sub = res_df[res_df['lead'] == lead].sort_values('week_start')
        x = range(len(sub))
        labels = [f"{r['week_start'].strftime('%b %d')}" for _, r in sub.iterrows()]

        ax.bar([xi - 0.2 for xi in x], sub['baseline_mae'], 0.2, label='Baseline', color='#e67e22', alpha=0.5)
        ax.bar([xi for xi in x], sub['static_mae'], 0.2, label='Static model', color='#e74c3c', alpha=0.7)
        ax.bar([xi + 0.2 for xi in x], sub['retrained_mae'], 0.2, label='Weekly retrained', color='#2ecc71', alpha=0.7)

        for xi, (_, row) in zip(x, sub.iterrows()):
            imp = row['improvement_pct']
            color = '#2ecc71' if imp > 0 else '#e74c3c'
            ax.text(xi + 0.2, row['retrained_mae'] + 0.1, f'{imp:+.1f}%',
                    ha='center', va='bottom', fontsize=8, color=color)

        ax.set_ylabel('MAE (MWh)')
        ax.set_title(f'Lead {lead} min')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Static vs Weekly Retrained Model', fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / '07_weekly_retrain_simulation.png', dpi=150)
    plt.close(fig)
    print("\n[+] 07_weekly_retrain_simulation.png saved")
    print("[+] data/weekly_retrain_sim.csv saved")


if __name__ == '__main__':
    main()
