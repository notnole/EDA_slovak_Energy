"""
Retrain Simulation - Does retraining help or is volatility inherent?

Strategy: Retrain whenever 7-day rolling MAE exceeds +20% of baseline.
Compare: Static 2024 model vs Adaptive retraining model.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

FEATURES_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\features")
MASTER_DIR = Path(r"C:\Users\20254757\pycharmprojects\ipesoft_eda_data\data\master")
OUTPUT_DIR = Path(__file__).parent


def load_data():
    print("[*] Loading data...")
    reg_df = pd.read_csv(FEATURES_DIR / 'regulation_3min.csv', parse_dates=['datetime'])
    load_df = pd.read_csv(FEATURES_DIR / 'load_3min.csv', parse_dates=['datetime'])
    label_df = pd.read_csv(MASTER_DIR / 'master_imbalance_data.csv', parse_dates=['datetime'])
    label_df = label_df[['datetime', 'System Imbalance (MWh)']].rename(
        columns={'System Imbalance (MWh)': 'imbalance'}
    )
    return reg_df, load_df, label_df


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


def add_proxy_features(df, train_end):
    """Add proxy and lag features, computing bias from training data."""
    df = df.sort_values('datetime').copy()

    reg_cols = ['reg_min0', 'reg_min3', 'reg_min6', 'reg_min9', 'reg_min12']
    available = [c for c in reg_cols if c in df.columns]
    df['period_proxy'] = -0.25 * df[available].mean(axis=1)

    # Lag features
    df['proxy_lag1'] = df['period_proxy'].shift(1)
    df['proxy_lag2'] = df['period_proxy'].shift(2)
    df['proxy_lag3'] = df['period_proxy'].shift(3)
    df['proxy_lag4'] = df['period_proxy'].shift(4)
    df['proxy_rolling_mean4'] = df['period_proxy'].shift(1).rolling(4).mean()
    df['proxy_rolling_std4'] = df['period_proxy'].shift(1).rolling(4).std()
    df['proxy_last_sign'] = np.sign(df['period_proxy'].shift(1))

    # Compute bias lookup from training data
    train_df = df[df['datetime'] < train_end].copy()
    train_df['proxy_error'] = train_df['period_proxy'] - train_df['imbalance']
    train_df['proxy_sign'] = np.sign(train_df['period_proxy'])

    bias_by_hour_sign = train_df.groupby(['hour', 'proxy_sign'])['proxy_error'].mean().to_dict()
    bias_by_hour = train_df.groupby('hour')['proxy_error'].mean().to_dict()

    # Apply bias
    df['proxy_sign_temp'] = np.sign(df['period_proxy'].shift(1))
    df['expected_proxy_bias'] = df.apply(
        lambda r: bias_by_hour_sign.get((r['hour'], r['proxy_sign_temp']),
                  bias_by_hour.get(r['hour'], 0)), axis=1
    )
    df['proxy_corrected'] = df['proxy_lag1'] - df['expected_proxy_bias']
    df = df.drop(columns=['proxy_sign_temp'])

    # Cycle features
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


def compute_lead_features(df, lead_time, train_end):
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
        result['baseline_pred'] = -0.25 * (0.4 * df['reg_min9'] + 0.2 * df['reg_min6'] + 0.2 * df['reg_min3'] + 0.2 * df['reg_min0'])
    else:
        result['baseline_pred'] = -0.25 * df[reg_cols].mean(axis=1)

    # Load deviation using training data expected
    if len(load_cols) > 0:
        train_load = df[df['datetime'] < train_end]
        expected_load = train_load.groupby(['hour', 'is_weekend'])[load_cols[0]].mean().to_dict()
        result['load_deviation'] = df[load_cols].mean(axis=1) - df.apply(
            lambda r: expected_load.get((r['hour'], r['is_weekend']), df[load_cols[0]].mean()), axis=1
        )
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


def get_feature_cols():
    """Feature columns for Lead 12."""
    return [
        'baseline_pred', 'reg_cumulative_mean',
        'reg_hist_mean_10', 'reg_hist_std_10', 'reg_hist_mean_20', 'reg_momentum',
        'proxy_lag1', 'proxy_lag2', 'proxy_lag3', 'proxy_lag4',
        'proxy_rolling_mean4', 'proxy_rolling_std4', 'proxy_last_sign',
        'hour_sin', 'hour_cos', 'is_weekend', 'load_deviation',
        'expected_proxy_bias', 'proxy_corrected',
        'cycle_30min', 'cycle_60min', 'cycle_pattern',
        'proxy_momentum', 'proxy_curvature', 'positive_regime', 'negative_regime',
    ]


def train_model(df, feature_cols):
    """Train LightGBM model."""
    X = df[feature_cols].values
    y = df['imbalance'].values
    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y) & ~np.isinf(X).any(axis=1)
    X, y = X[valid], y[valid]

    if len(X) < 100:
        return None

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


def predict(model, df, feature_cols):
    """Get predictions."""
    X = df[feature_cols].values
    valid = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
    preds = np.full(len(df), np.nan)
    if valid.sum() > 0:
        preds[valid] = model.predict(X[valid])
    return preds


def main():
    print("=" * 70)
    print("RETRAIN SIMULATION")
    print("Does retraining help, or is volatility inherent?")
    print("=" * 70)

    reg_df, load_df, label_df = load_data()
    df_base = create_base_features(reg_df, load_df, label_df)

    initial_train_end = pd.Timestamp('2025-01-01')
    test_start = pd.Timestamp('2025-01-01')
    lead = 12
    feature_cols = get_feature_cols()

    # Threshold for retraining: +20% above current model's training MAE
    RETRAIN_THRESHOLD = 0.20
    MIN_DAYS_BETWEEN_RETRAIN = 14  # Don't retrain more often than every 2 weeks

    # ========================================================================
    # Strategy 1: Static model (train on 2024 only, never retrain)
    # ========================================================================
    print("\n[*] Strategy 1: Static model (2024 only, no retraining)...")

    df_static = add_proxy_features(df_base.copy(), initial_train_end)
    df_static = compute_lead_features(df_static, lead, initial_train_end)

    train_static = df_static[df_static['datetime'] < initial_train_end]
    test_static = df_static[df_static['datetime'] >= test_start].copy()

    feature_cols_avail = [c for c in feature_cols if c in df_static.columns]
    model_static = train_model(train_static, feature_cols_avail)

    test_static['pred_static'] = predict(model_static, test_static, feature_cols_avail)
    test_static['error_static'] = np.abs(test_static['pred_static'] - test_static['imbalance'])

    static_mae = test_static['error_static'].mean()
    print(f"    Static model MAE: {static_mae:.3f} MWh")

    # ========================================================================
    # Strategy 2: Adaptive retraining (retrain when trigger fires)
    # ========================================================================
    print("\n[*] Strategy 2: Adaptive retraining (retrain on +20% trigger)...")

    test_dates = test_static['datetime'].dt.date.unique()
    test_dates = sorted(test_dates)

    # Initialize
    current_train_end = initial_train_end
    current_model = model_static
    last_retrain_date = initial_train_end.date()

    # Track training MAE for threshold calculation
    train_data = df_static[df_static['datetime'] < current_train_end]
    X_train = train_data[feature_cols_avail].dropna()
    y_train = train_data.loc[X_train.index, 'imbalance']
    train_preds = current_model.predict(X_train.values)
    current_baseline_mae = np.abs(train_preds - y_train.values).mean()

    retrain_dates = []
    predictions_adaptive = []
    rolling_mae_history = []

    # Process day by day
    print("    Simulating day-by-day with retrain triggers...")

    for i, date in enumerate(test_dates):
        date_ts = pd.Timestamp(date)
        day_data = test_static[test_static['datetime'].dt.date == date].copy()

        if len(day_data) == 0:
            continue

        # Get predictions with current model
        day_preds = predict(current_model, day_data, feature_cols_avail)
        day_errors = np.abs(day_preds - day_data['imbalance'].values)
        day_mae = np.nanmean(day_errors)

        # Store predictions
        for idx, pred, err in zip(day_data.index, day_preds, day_errors):
            predictions_adaptive.append({
                'datetime': day_data.loc[idx, 'datetime'],
                'imbalance': day_data.loc[idx, 'imbalance'],
                'pred_adaptive': pred,
                'error_adaptive': err,
                'model_train_end': current_train_end,
            })

        # Calculate 7-day rolling MAE
        recent_errors = [p['error_adaptive'] for p in predictions_adaptive[-96*7:]]  # ~7 days of 15-min data
        rolling_mae = np.nanmean(recent_errors) if len(recent_errors) > 96 else day_mae
        rolling_mae_history.append({'date': date, 'rolling_mae': rolling_mae, 'baseline': current_baseline_mae})

        # Check retrain trigger
        days_since_retrain = (date - last_retrain_date).days
        pct_above = (rolling_mae / current_baseline_mae - 1)

        if pct_above > RETRAIN_THRESHOLD and days_since_retrain >= MIN_DAYS_BETWEEN_RETRAIN:
            print(f"    [{date}] Trigger fired: rolling MAE {rolling_mae:.2f} is +{pct_above*100:.0f}% above baseline {current_baseline_mae:.2f}")

            # Retrain on all data up to this point
            new_train_end = date_ts

            # Recompute features with new training period
            df_retrain = add_proxy_features(df_base.copy(), new_train_end)
            df_retrain = compute_lead_features(df_retrain, lead, new_train_end)

            train_retrain = df_retrain[df_retrain['datetime'] < new_train_end]
            new_model = train_model(train_retrain, feature_cols_avail)

            if new_model is not None:
                current_model = new_model
                current_train_end = new_train_end
                last_retrain_date = date
                retrain_dates.append(date)

                # Update baseline MAE
                X_new = train_retrain[feature_cols_avail].dropna()
                y_new = train_retrain.loc[X_new.index, 'imbalance']
                new_preds = current_model.predict(X_new.values)
                current_baseline_mae = np.abs(new_preds - y_new.values).mean()

                print(f"             Retrained. New baseline MAE: {current_baseline_mae:.2f} MWh")

    # Convert to DataFrame
    df_adaptive = pd.DataFrame(predictions_adaptive)
    adaptive_mae = df_adaptive['error_adaptive'].mean()

    print(f"\n    Adaptive model MAE: {adaptive_mae:.3f} MWh")
    print(f"    Number of retrains: {len(retrain_dates)}")
    if retrain_dates:
        print(f"    Retrain dates: {retrain_dates}")

    # ========================================================================
    # Compare results
    # ========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    improvement = (static_mae - adaptive_mae) / static_mae * 100
    print(f"\n  Static MAE:   {static_mae:.3f} MWh")
    print(f"  Adaptive MAE: {adaptive_mae:.3f} MWh")
    print(f"  Improvement:  {improvement:+.1f}%")

    if improvement > 5:
        conclusion = "RETRAINING HELPS - degradation is due to stale model"
    elif improvement < -5:
        conclusion = "RETRAINING HURTS - model is overfitting to recent noise"
    else:
        conclusion = "RETRAINING HAS MINIMAL IMPACT - volatility is inherent"

    print(f"\n  CONCLUSION: {conclusion}")

    # ========================================================================
    # Monthly comparison
    # ========================================================================
    test_static['date'] = test_static['datetime'].dt.date
    df_adaptive['date'] = df_adaptive['datetime'].dt.date

    # Merge for comparison
    merged = test_static[['datetime', 'imbalance', 'error_static']].merge(
        df_adaptive[['datetime', 'error_adaptive', 'model_train_end']],
        on='datetime', how='inner'
    )
    merged['month'] = merged['datetime'].dt.to_period('M')

    monthly = merged.groupby('month').agg({
        'error_static': 'mean',
        'error_adaptive': 'mean',
    }).reset_index()
    monthly.columns = ['month', 'mae_static', 'mae_adaptive']
    monthly['improvement'] = (monthly['mae_static'] - monthly['mae_adaptive']) / monthly['mae_static'] * 100

    print("\n  Monthly Comparison:")
    print("  " + "-" * 55)
    print(f"  {'Month':<10} {'Static':<12} {'Adaptive':<12} {'Improvement':<12}")
    print("  " + "-" * 55)
    for _, row in monthly.iterrows():
        print(f"  {str(row['month']):<10} {row['mae_static']:<12.2f} {row['mae_adaptive']:<12.2f} {row['improvement']:>+10.1f}%")

    # ========================================================================
    # Visualization
    # ========================================================================
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Plot 1: Daily MAE comparison
    ax1 = axes[0]
    daily_static = test_static.groupby('date')['error_static'].mean()
    daily_adaptive = df_adaptive.groupby('date')['error_adaptive'].mean()

    dates = pd.to_datetime(daily_static.index)
    ax1.plot(dates, daily_static.rolling(7).mean(), label='Static (2024 only)', color='red', alpha=0.8)
    ax1.plot(dates, daily_adaptive.rolling(7).mean(), label='Adaptive (retrain on trigger)', color='green', alpha=0.8)

    # Mark retrain dates
    for rd in retrain_dates:
        ax1.axvline(pd.Timestamp(rd), color='blue', linestyle='--', alpha=0.5, linewidth=1)

    ax1.set_ylabel('7-day Rolling MAE (MWh)')
    ax1.set_title('Static vs Adaptive Model: 7-day Rolling MAE')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    # Add retrain annotation
    if retrain_dates:
        ax1.annotate(f'{len(retrain_dates)} retrains', xy=(0.02, 0.95), xycoords='axes fraction',
                    fontsize=10, color='blue')

    # Plot 2: Monthly bars
    ax2 = axes[1]
    x = np.arange(len(monthly))
    width = 0.35

    bars1 = ax2.bar(x - width/2, monthly['mae_static'], width, label='Static', color='red', alpha=0.7)
    bars2 = ax2.bar(x + width/2, monthly['mae_adaptive'], width, label='Adaptive', color='green', alpha=0.7)

    ax2.set_ylabel('MAE (MWh)')
    ax2.set_title('Monthly MAE: Static vs Adaptive')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(m) for m in monthly['month']], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Add improvement labels
    for i, imp in enumerate(monthly['improvement']):
        color = 'green' if imp > 0 else 'red'
        ax2.annotate(f'{imp:+.0f}%', xy=(i, max(monthly['mae_static'].iloc[i], monthly['mae_adaptive'].iloc[i]) + 0.1),
                    ha='center', fontsize=8, color=color)

    # Plot 3: Cumulative error difference
    ax3 = axes[2]
    merged_sorted = merged.sort_values('datetime')
    merged_sorted['cum_diff'] = (merged_sorted['error_static'] - merged_sorted['error_adaptive']).cumsum()

    ax3.fill_between(merged_sorted['datetime'], 0, merged_sorted['cum_diff'],
                     where=merged_sorted['cum_diff'] > 0, color='green', alpha=0.5, label='Adaptive better')
    ax3.fill_between(merged_sorted['datetime'], 0, merged_sorted['cum_diff'],
                     where=merged_sorted['cum_diff'] < 0, color='red', alpha=0.5, label='Static better')
    ax3.axhline(0, color='black', linewidth=1)

    for rd in retrain_dates:
        ax3.axvline(pd.Timestamp(rd), color='blue', linestyle='--', alpha=0.5)

    ax3.set_ylabel('Cumulative Error Difference (MWh)')
    ax3.set_xlabel('Date')
    ax3.set_title('Cumulative Advantage: Positive = Adaptive wins, Negative = Static wins')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_retrain_simulation.png', dpi=150)
    plt.close()
    print("\n[+] Saved 04_retrain_simulation.png")

    # ========================================================================
    # Summary stats
    # ========================================================================
    summary = {
        'static_mae': static_mae,
        'adaptive_mae': adaptive_mae,
        'improvement_pct': improvement,
        'num_retrains': len(retrain_dates),
        'retrain_dates': [str(d) for d in retrain_dates],
        'conclusion': conclusion,
    }

    # Save results
    results_df = merged[['datetime', 'imbalance', 'error_static', 'error_adaptive', 'model_train_end']]
    results_df.to_csv(OUTPUT_DIR / 'retrain_simulation_results.csv', index=False)
    print("[+] Saved retrain_simulation_results.csv")

    return summary, monthly


if __name__ == '__main__':
    summary, monthly = main()
