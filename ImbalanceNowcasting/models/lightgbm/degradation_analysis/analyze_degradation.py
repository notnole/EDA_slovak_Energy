"""
Model Degradation Analysis - Train on 2024, Test on 2025+

Analyzes when model performance degrades and when retraining is needed.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

FEATURES_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\features")
MASTER_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\master")
OUTPUT_DIR = Path(__file__).parent

# Use same functions as v7 for consistency
def load_data():
    print("[*] Loading data...")
    reg_df = pd.read_csv(FEATURES_DIR / 'regulation_3min.csv', parse_dates=['datetime'])
    load_df = pd.read_csv(FEATURES_DIR / 'load_3min.csv', parse_dates=['datetime'])
    label_df = pd.read_csv(MASTER_DIR / 'master_imbalance_data.csv', parse_dates=['datetime'])
    label_df = label_df[['datetime', 'System Imbalance (MWh)']].rename(
        columns={'System Imbalance (MWh)': 'imbalance'}
    )
    return reg_df, load_df, label_df


def compute_load_expected(load_df, train_end):
    load_df = load_df.copy()
    load_df['hour'] = load_df['datetime'].dt.hour
    load_df['minute'] = load_df['datetime'].dt.minute
    load_df['is_weekend'] = load_df['datetime'].dt.dayofweek >= 5
    train_mask = load_df['datetime'] < train_end
    expected = load_df[train_mask].groupby(['hour', 'minute', 'is_weekend'])['load_mw'].mean()
    expected.name = 'expected_load'
    return expected


def add_3min_features(reg_df):
    reg_df = reg_df.sort_values('datetime').copy()
    reg_df['reg_hist_mean_10'] = reg_df['regulation_mw'].shift(1).rolling(10).mean()
    reg_df['reg_hist_std_10'] = reg_df['regulation_mw'].shift(1).rolling(10).std()
    reg_df['reg_hist_mean_20'] = reg_df['regulation_mw'].shift(1).rolling(20).mean()
    reg_df['reg_momentum'] = reg_df['regulation_mw'].shift(1) - reg_df['regulation_mw'].shift(2)
    return reg_df


def create_base_features(reg_df, load_df, label_df):
    print("[*] Creating base features...")
    reg_df = add_3min_features(reg_df)

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

    hist_cols = ['reg_hist_mean_10', 'reg_hist_std_10', 'reg_hist_mean_20', 'reg_momentum']
    reg_min0 = reg_df[reg_df['minute_in_qh'] == 0][['settlement_start'] + hist_cols].copy()
    reg_min0 = reg_min0.rename(columns={'settlement_start': 'datetime'})

    df = pd.merge(label_df, pivot_reg, on='datetime', how='inner')
    df = pd.merge(df, reg_min0, on='datetime', how='left')

    # Load
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

    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    return df


def compute_proxy_bias_lookup(df, train_end):
    train_df = df[df['datetime'] < train_end].copy()
    reg_cols = ['reg_min0', 'reg_min3', 'reg_min6', 'reg_min9', 'reg_min12']
    available = [c for c in reg_cols if c in train_df.columns]
    train_df['proxy'] = -0.25 * train_df[available].mean(axis=1)
    train_df['proxy_error'] = train_df['proxy'] - train_df['imbalance']
    train_df['proxy_sign'] = np.sign(train_df['proxy'])

    bias_by_hour_sign = train_df.groupby(['hour', 'proxy_sign'])['proxy_error'].agg(['mean', 'std']).reset_index()
    bias_by_hour_sign.columns = ['hour', 'proxy_sign', 'expected_bias', 'bias_std']
    bias_by_hour = train_df.groupby('hour')['proxy_error'].mean().reset_index()
    bias_by_hour.columns = ['hour', 'expected_bias_hour']

    return bias_by_hour_sign, bias_by_hour


def add_smart_error_features(df, bias_by_hour_sign, bias_by_hour):
    df = df.sort_values('datetime').copy()

    reg_cols = ['reg_min0', 'reg_min3', 'reg_min6', 'reg_min9', 'reg_min12']
    available = [c for c in reg_cols if c in df.columns]
    df['period_proxy'] = -0.25 * df[available].mean(axis=1)

    df['proxy_lag1'] = df['period_proxy'].shift(1)
    df['proxy_lag2'] = df['period_proxy'].shift(2)
    df['proxy_lag3'] = df['period_proxy'].shift(3)
    df['proxy_lag4'] = df['period_proxy'].shift(4)
    df['proxy_rolling_mean4'] = df['period_proxy'].shift(1).rolling(4).mean()
    df['proxy_rolling_std4'] = df['period_proxy'].shift(1).rolling(4).std()
    df['proxy_last_sign'] = np.sign(df['period_proxy'].shift(1))

    df['proxy_sign'] = np.sign(df['period_proxy'].shift(1))
    df = df.merge(bias_by_hour_sign, on=['hour', 'proxy_sign'], how='left')
    df = df.merge(bias_by_hour, on='hour', how='left')
    df['expected_proxy_bias'] = df['expected_bias'].fillna(df['expected_bias_hour'])
    df['proxy_corrected'] = df['proxy_lag1'] - df['expected_proxy_bias']
    df = df.drop(columns=['proxy_sign', 'expected_bias', 'bias_std', 'expected_bias_hour'], errors='ignore')

    df['cycle_30min'] = df['proxy_lag1'] - df['proxy_lag2']
    df['cycle_60min'] = df['proxy_lag1'] - df['proxy_lag4']
    df['cycle_pattern'] = (df['proxy_lag2'] + df['proxy_lag4']) / 2 - df['proxy_lag1']
    df['proxy_momentum'] = df['proxy_lag1'] - df['proxy_lag2']
    df['proxy_curvature'] = df['proxy_lag1'] - 2*df['proxy_lag2'] + df['proxy_lag3']

    positive = (df['period_proxy'] > 0).astype(float)
    df['proxy_prop_positive_4'] = positive.shift(1).rolling(4).mean()
    df['positive_regime'] = (df['proxy_prop_positive_4'] > 0.75).astype(int)
    df['negative_regime'] = (df['proxy_prop_positive_4'] < 0.25).astype(int)

    return df


def compute_lead_features(df, lead_time, load_expected):
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
        result['baseline_pred'] = -0.25 * (0.4 * df['reg_min9'] + 0.2 * df['reg_min6'] + 0.2 * df['reg_min3'] + 0.2 * df['reg_min0'])
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
    else:
        result['reg_std'] = 0
        result['reg_range'] = 0
        result['reg_trend'] = 0

    return result


def get_features_for_lead(lead_time):
    core = ['baseline_pred', 'reg_cumulative_mean']
    within_period = ['reg_std', 'reg_range', 'reg_trend']

    if lead_time == 12:
        return core + [
            'reg_hist_mean_10', 'reg_hist_std_10', 'reg_hist_mean_20', 'reg_momentum',
            'proxy_lag1', 'proxy_lag2', 'proxy_lag3', 'proxy_lag4',
            'proxy_rolling_mean4', 'proxy_rolling_std4', 'proxy_last_sign',
            'hour_sin', 'hour_cos', 'is_weekend', 'load_deviation',
            'expected_proxy_bias', 'proxy_corrected',
            'cycle_30min', 'cycle_60min', 'cycle_pattern',
            'proxy_momentum', 'proxy_curvature', 'positive_regime', 'negative_regime',
        ]
    elif lead_time == 9:
        return core + within_period + [
            'reg_hist_mean_10', 'reg_hist_mean_20',
            'proxy_lag1', 'proxy_lag2', 'proxy_lag3',
            'proxy_rolling_mean4', 'proxy_rolling_std4', 'proxy_last_sign',
            'hour_sin', 'hour_cos', 'is_weekend', 'load_deviation',
            'expected_proxy_bias', 'proxy_corrected',
            'cycle_30min', 'cycle_60min', 'proxy_momentum',
        ]
    elif lead_time == 6:
        return core + within_period + [
            'reg_hist_mean_10', 'proxy_lag1', 'proxy_lag2', 'proxy_rolling_mean4',
            'hour_sin', 'hour_cos', 'expected_proxy_bias', 'cycle_30min',
        ]
    elif lead_time == 3:
        return core + within_period + [
            'proxy_lag1', 'proxy_rolling_mean4', 'hour_sin', 'hour_cos',
        ]
    else:
        return core + within_period + ['proxy_lag1']


def train_model(train_df, feature_cols):
    X = train_df[feature_cols].values
    y = train_df['imbalance'].values
    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y) & ~np.isinf(X).any(axis=1)
    X, y = X[valid], y[valid]

    params = {
        'objective': 'regression', 'metric': 'mae', 'boosting_type': 'gbdt',
        'num_leaves': 31, 'max_depth': 6, 'learning_rate': 0.05,
        'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
        'min_data_in_leaf': 50, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
        'verbose': -1, 'seed': 42,
    }

    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(params, train_data, num_boost_round=500)
    return model


def main():
    print("=" * 70)
    print("MODEL DEGRADATION ANALYSIS")
    print("Train: 2024 only | Test: 2025+")
    print("=" * 70)

    reg_df, load_df, label_df = load_data()

    train_end = pd.Timestamp('2025-01-01')
    load_expected = compute_load_expected(load_df, train_end)
    df = create_base_features(reg_df, load_df, label_df)

    print("[*] Computing proxy bias lookup...")
    bias_by_hour_sign, bias_by_hour = compute_proxy_bias_lookup(df, train_end)
    df = add_smart_error_features(df, bias_by_hour_sign, bias_by_hour)

    # Focus on Lead 12 (hardest, most sensitive to drift)
    lead = 12
    print(f"\n[*] Training model for Lead {lead} min...")

    lead_df = compute_lead_features(df, lead, load_expected)
    train_df = lead_df[lead_df['datetime'] < train_end]
    test_df = lead_df[lead_df['datetime'] >= train_end].copy()

    feature_cols = get_features_for_lead(lead)
    feature_cols = [c for c in feature_cols if c in lead_df.columns]

    print(f"    Train: {len(train_df):,} | Test: {len(test_df):,}")

    model = train_model(train_df, feature_cols)

    # Get predictions
    X_test = test_df[feature_cols].values
    valid_mask = ~np.isnan(X_test).any(axis=1)
    test_df = test_df[valid_mask].copy()
    X_test = X_test[valid_mask]

    test_df['prediction'] = model.predict(X_test)
    test_df['error'] = test_df['prediction'] - test_df['imbalance']
    test_df['abs_error'] = np.abs(test_df['error'])
    test_df['date'] = test_df['datetime'].dt.date

    # Daily metrics
    daily = test_df.groupby('date').agg({
        'abs_error': ['mean', 'std', 'count'],
        'imbalance': 'std',
        'error': 'mean'
    }).reset_index()
    daily.columns = ['date', 'daily_mae', 'daily_mae_std', 'n_samples', 'imbalance_std', 'daily_bias']
    daily['date'] = pd.to_datetime(daily['date'])

    # Standardized MAE (MAE / daily imbalance std)
    daily['mae_standardized'] = daily['daily_mae'] / daily['imbalance_std']

    # Rolling metrics (7-day and 30-day)
    daily['mae_roll_7d'] = daily['daily_mae'].rolling(7, min_periods=3).mean()
    daily['mae_roll_30d'] = daily['daily_mae'].rolling(30, min_periods=7).mean()

    # Training period baseline MAE
    train_mae = train_df[feature_cols + ['imbalance']].dropna()
    X_train = train_mae[feature_cols].values
    y_train = train_mae['imbalance'].values
    train_pred = model.predict(X_train)
    baseline_mae = np.abs(train_pred - y_train).mean()
    print(f"    Training MAE (baseline): {baseline_mae:.3f} MWh")

    # Thresholds
    threshold_10pct = baseline_mae * 1.10
    threshold_20pct = baseline_mae * 1.20
    threshold_30pct = baseline_mae * 1.30

    daily['above_10pct'] = daily['mae_roll_7d'] > threshold_10pct
    daily['above_20pct'] = daily['mae_roll_7d'] > threshold_20pct
    daily['above_30pct'] = daily['mae_roll_7d'] > threshold_30pct

    # CUSUM for drift detection
    daily['mae_deviation'] = daily['daily_mae'] - baseline_mae
    daily['cusum'] = daily['mae_deviation'].cumsum()

    # Save daily data
    daily.to_csv(OUTPUT_DIR / 'daily_performance.csv', index=False)
    print(f"[+] Saved daily_performance.csv")

    # ========================================================================
    # PLOT 1: Daily MAE with rolling average and thresholds
    # ========================================================================
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.scatter(daily['date'], daily['daily_mae'], alpha=0.3, s=10, color='gray', label='Daily MAE')
    ax.plot(daily['date'], daily['mae_roll_7d'], color='blue', linewidth=1.5, label='7-day rolling MAE')
    ax.plot(daily['date'], daily['mae_roll_30d'], color='darkblue', linewidth=2, label='30-day rolling MAE')

    ax.axhline(baseline_mae, color='green', linestyle='-', linewidth=2, label=f'Training baseline ({baseline_mae:.2f})')
    ax.axhline(threshold_10pct, color='orange', linestyle='--', label=f'+10% threshold ({threshold_10pct:.2f})')
    ax.axhline(threshold_20pct, color='red', linestyle='--', label=f'+20% threshold ({threshold_20pct:.2f})')

    # Shade regions above threshold
    ax.fill_between(daily['date'], threshold_20pct, daily['mae_roll_7d'].clip(lower=threshold_20pct),
                    alpha=0.3, color='red', label='Above +20%')

    ax.set_xlabel('Date')
    ax.set_ylabel('MAE (MWh)')
    ax.set_title(f'Lead {lead} min: Daily MAE Over Time (Trained on 2024 only)')
    ax.legend(loc='upper left', fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, daily['daily_mae'].quantile(0.98) * 1.1)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_daily_mae_over_time.png', dpi=150)
    plt.close()
    print("[+] Saved 01_daily_mae_over_time.png")

    # ========================================================================
    # PLOT 2: CUSUM chart for drift detection
    # ========================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top: CUSUM
    ax1.plot(daily['date'], daily['cusum'], color='purple', linewidth=2)
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax1.fill_between(daily['date'], 0, daily['cusum'], where=daily['cusum'] > 0,
                     alpha=0.3, color='red', label='Cumulative over-error')
    ax1.fill_between(daily['date'], 0, daily['cusum'], where=daily['cusum'] < 0,
                     alpha=0.3, color='green', label='Cumulative under-error')
    ax1.set_ylabel('CUSUM (MWh)')
    ax1.set_title(f'Lead {lead} min: CUSUM Drift Detection - Retrain when slope stays positive')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Bottom: Daily deviation
    colors = ['red' if x > 0 else 'green' for x in daily['mae_deviation']]
    ax2.bar(daily['date'], daily['mae_deviation'], color=colors, alpha=0.6, width=1)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('MAE - Baseline (MWh)')
    ax2.set_title('Daily deviation from training baseline')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_cusum_drift_detection.png', dpi=150)
    plt.close()
    print("[+] Saved 02_cusum_drift_detection.png")

    # ========================================================================
    # PLOT 3: Monthly performance summary with retrain recommendation
    # ========================================================================
    test_df['month'] = test_df['datetime'].dt.to_period('M')
    monthly = test_df.groupby('month').agg({
        'abs_error': 'mean',
        'error': ['mean', 'std'],
        'imbalance': 'std'
    }).reset_index()
    monthly.columns = ['month', 'mae', 'bias', 'error_std', 'imbalance_std']
    monthly['month_str'] = monthly['month'].astype(str)
    monthly['pct_vs_baseline'] = (monthly['mae'] / baseline_mae - 1) * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Top: Monthly MAE bars
    colors = ['green' if x < 10 else 'orange' if x < 20 else 'red' for x in monthly['pct_vs_baseline']]
    bars = ax1.bar(monthly['month_str'], monthly['mae'], color=colors, edgecolor='black', alpha=0.8)
    ax1.axhline(baseline_mae, color='green', linestyle='-', linewidth=2, label=f'Training baseline ({baseline_mae:.2f})')
    ax1.axhline(threshold_20pct, color='red', linestyle='--', linewidth=2, label=f'+20% retrain trigger')

    # Add percentage labels
    for bar, pct in zip(bars, monthly['pct_vs_baseline']):
        height = bar.get_height()
        ax1.annotate(f'{pct:+.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

    ax1.set_ylabel('MAE (MWh)')
    ax1.set_title(f'Lead {lead} min: Monthly MAE vs Training Baseline')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')

    # Bottom: Retrain recommendation timeline
    ax2.set_xlim(-0.5, len(monthly) - 0.5)
    ax2.set_ylim(0, 1)

    for i, (month, pct) in enumerate(zip(monthly['month_str'], monthly['pct_vs_baseline'])):
        if pct < 10:
            color, status = 'green', 'OK'
        elif pct < 20:
            color, status = 'orange', 'MONITOR'
        else:
            color, status = 'red', 'RETRAIN'

        ax2.add_patch(plt.Rectangle((i - 0.4, 0.1), 0.8, 0.8, color=color, alpha=0.7))
        ax2.text(i, 0.5, status, ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    ax2.set_xticks(range(len(monthly)))
    ax2.set_xticklabels(monthly['month_str'], rotation=45, ha='right')
    ax2.set_yticks([])
    ax2.set_title('Retrain Recommendation: GREEN=OK (<10%), ORANGE=Monitor (10-20%), RED=Retrain (>20%)')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_monthly_retrain_recommendation.png', dpi=150)
    plt.close()
    print("[+] Saved 03_monthly_retrain_recommendation.png")

    # ========================================================================
    # Summary statistics
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTraining baseline MAE: {baseline_mae:.3f} MWh")
    print(f"\nMonthly Performance (Lead {lead} min):")
    print("-" * 50)
    for _, row in monthly.iterrows():
        status = 'OK' if row['pct_vs_baseline'] < 10 else 'MONITOR' if row['pct_vs_baseline'] < 20 else 'RETRAIN'
        print(f"  {row['month_str']}: MAE={row['mae']:.2f} MWh ({row['pct_vs_baseline']:+.1f}%) [{status}]")

    # Days above threshold
    days_above_10 = daily['above_10pct'].sum()
    days_above_20 = daily['above_20pct'].sum()
    total_days = len(daily)

    print(f"\nDays above +10% threshold: {days_above_10}/{total_days} ({100*days_above_10/total_days:.1f}%)")
    print(f"Days above +20% threshold: {days_above_20}/{total_days} ({100*days_above_20/total_days:.1f}%)")

    # First day above thresholds
    if days_above_20 > 0:
        first_above_20 = daily[daily['above_20pct']]['date'].iloc[0]
        print(f"\nFirst day above +20%: {first_above_20.strftime('%Y-%m-%d')}")
        days_until_retrain = (first_above_20 - train_end).days
        print(f"Days after training cutoff: {days_until_retrain}")

    return daily, monthly


if __name__ == '__main__':
    daily, monthly = main()
