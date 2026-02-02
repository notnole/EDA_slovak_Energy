"""
Plot Tricky Days Analysis
=========================
Visualize forecasts vs actual 3-min load on difficult days
to find hidden patterns and remaining signal.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = Path(__file__).parent.parent.parent.parent.parent  # ipesoft_eda_data
OUTPUT_PATH = Path(__file__).parent / 'plots'
OUTPUT_PATH.mkdir(exist_ok=True)


def load_data():
    """Load all data."""
    print("Loading data...")

    # Hourly data
    df = pd.read_parquet(BASE_PATH / 'features' / 'DamasLoad' / 'load_data.parquet')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df['year'] = df['datetime'].dt.year
    df['hour'] = df['datetime'].dt.hour
    df['dow'] = df['datetime'].dt.dayofweek
    df['error'] = df['actual_load_mw'] - df['forecast_load_mw']

    # 3-minute data
    load_3min = pd.read_csv(BASE_PATH / 'data' / 'features' / 'load_3min.csv')
    load_3min['datetime'] = pd.to_datetime(load_3min['datetime'])
    load_3min = load_3min.sort_values('datetime')

    print(f"  Loaded {len(df):,} hourly records")
    print(f"  Loaded {len(load_3min):,} 3-min records")

    return df, load_3min


def train_two_stage_models(df):
    """Train two-stage models and get predictions."""
    print("\nTraining models...")

    # Create features
    for lag in range(1, 7):
        df[f'error_lag{lag}'] = df['error'].shift(lag)
    df['error_roll_mean_6h'] = df['error'].shift(1).rolling(6).mean()

    for h in range(1, 6):
        df[f'target_h{h}'] = df['error'].shift(-h)

    features = ['error_lag1', 'error_lag2', 'error_lag3', 'error_lag4',
                'error_roll_mean_6h', 'hour', 'dow']

    df_model = df.dropna(subset=['target_h1'] + features).copy()
    train = df_model[df_model['year'] == 2024]
    test = df_model[df_model['year'] >= 2025].copy()

    # Stage 1 for each horizon
    for h in range(1, 6):
        target = f'target_h{h}'
        model = lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.03, max_depth=6,
            random_state=42, verbosity=-1
        )
        model.fit(train[features], train[target])
        test[f's1_pred_h{h}'] = model.predict(test[features])
        test[f's1_residual_h{h}'] = test[target] - test[f's1_pred_h{h}']

    # Stage 2 for each horizon
    for h in range(1, 6):
        for lag in range(1, 4):
            test[f'resid_lag{lag}_h{h}'] = test[f's1_residual_h{h}'].shift(lag)

        resid_features = [f'resid_lag1_h{h}', f'resid_lag2_h{h}']
        train_s2 = test.iloc[:len(test)//3].dropna(subset=resid_features)

        model_s2 = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=4,
            random_state=42, verbosity=-1
        )
        model_s2.fit(train_s2[resid_features], train_s2[f's1_residual_h{h}'])

        test_valid = test.dropna(subset=resid_features)
        test.loc[test_valid.index, f's2_pred_h{h}'] = (
            test_valid[f's1_pred_h{h}'] + model_s2.predict(test_valid[resid_features])
        )
        test[f'final_error_h{h}'] = test[f'target_h{h}'] - test[f's2_pred_h{h}']

    return test


def plot_tricky_day(date, test_df, load_3min_df, output_path):
    """Plot a single day with all forecasts vs actual 3-min load."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))

    # Filter data for this day
    day_hourly = test_df[test_df['date'] == date].copy()

    # Get 3-min data for this day (with buffer for next day predictions)
    start = pd.Timestamp(date)
    end = start + pd.Timedelta(days=1)
    day_3min = load_3min_df[
        (load_3min_df['datetime'] >= start - pd.Timedelta(hours=1)) &
        (load_3min_df['datetime'] < end + pd.Timedelta(hours=6))
    ]

    if len(day_hourly) == 0 or len(day_3min) == 0:
        return None

    # Calculate daily statistics
    damas_mae = np.abs(day_hourly['error']).mean()
    our_mae = np.abs(day_hourly['final_error_h1']).mean()

    # === PLOT 1: 3-min actual vs hourly forecasts ===
    ax1 = axes[0]

    # 3-minute actual load
    ax1.plot(day_3min['datetime'], day_3min['load_mw'],
             'b-', alpha=0.6, linewidth=1, label='Actual Load (3-min)')

    # Hourly actual
    ax1.step(day_hourly['datetime'], day_hourly['actual_load_mw'],
             'b-', where='post', linewidth=2.5, label='Actual Load (hourly avg)')

    # DAMAS forecast
    ax1.step(day_hourly['datetime'], day_hourly['forecast_load_mw'],
             'r--', where='post', linewidth=2, label='DAMAS Forecast')

    # Our forecasts at each horizon
    colors = {'H+1': 'green', 'H+3': 'orange', 'H+5': 'purple'}
    for h, color in [(1, 'green'), (3, 'orange'), (5, 'purple')]:
        if f's2_pred_h{h}' in day_hourly.columns:
            # Our prediction of error + DAMAS forecast = our load forecast
            corrected = day_hourly['forecast_load_mw'] + day_hourly[f's2_pred_h{h}']
            ax1.step(day_hourly['datetime'], corrected,
                    color=color, where='post', linewidth=1.5,
                    label=f'Our H+{h} Forecast', alpha=0.8)

    ax1.set_ylabel('Load (MW)', fontsize=12)
    ax1.set_title(f'Load Forecasts vs 3-min Actual - {date}\n'
                  f'DAMAS MAE: {damas_mae:.1f} MW | Our H+1 MAE: {our_mae:.1f} MW',
                  fontsize=14)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # === PLOT 2: Errors comparison ===
    ax2 = axes[1]
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # DAMAS error
    width = 0.025
    ax2.bar(day_hourly['datetime'], day_hourly['error'],
            width=width, alpha=0.4, label='DAMAS Error', color='red')

    # Our errors at each horizon
    for h, color in [(1, 'green'), (2, 'blue'), (3, 'orange'), (4, 'cyan'), (5, 'purple')]:
        if f'final_error_h{h}' in day_hourly.columns:
            offset = pd.Timedelta(minutes=(h-3)*10)
            ax2.bar(day_hourly['datetime'] + offset,
                   day_hourly[f'final_error_h{h}'],
                   width=width*0.7, alpha=0.6,
                   label=f'Our H+{h} Error', color=color)

    ax2.set_ylabel('Forecast Error (MW)', fontsize=12)
    ax2.set_title('Forecast Errors: DAMAS vs Our Model (all horizons)', fontsize=14)
    ax2.legend(loc='upper right', ncol=3, fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # === PLOT 3: Error pattern by hour ===
    ax3 = axes[2]

    hours = day_hourly['hour'].values

    # DAMAS error pattern
    ax3.plot(hours, day_hourly['error'].values, 'ro-',
             label='DAMAS Error', alpha=0.5, markersize=8, linewidth=2)

    # Our error pattern at each horizon
    for h, color in [(1, 'green'), (3, 'orange'), (5, 'purple')]:
        if f'final_error_h{h}' in day_hourly.columns:
            errors = day_hourly[f'final_error_h{h}'].values
            ax3.plot(hours, errors, 'o-', color=color,
                    label=f'Our H+{h} Error', alpha=0.7, markersize=5)

    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Hour of Day', fontsize=12)
    ax3.set_ylabel('Error (MW)', fontsize=12)
    ax3.set_title('Error Pattern Throughout the Day', fontsize=14)
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(range(0, 24, 2))
    ax3.set_xlim(-0.5, 23.5)

    plt.tight_layout()
    plt.savefig(output_path / f'tricky_day_{date}.png', dpi=150, bbox_inches='tight')
    plt.close()

    return True


def analyze_error_patterns(test_df):
    """Analyze when and why errors occur."""
    print("\n" + "=" * 60)
    print("ERROR PATTERN ANALYSIS")
    print("=" * 60)

    test_df['date'] = test_df['datetime'].dt.date
    test_df['abs_error_h1'] = np.abs(test_df['final_error_h1'])

    # By hour
    print("\nMAE by Hour of Day:")
    by_hour = test_df.groupby('hour')['abs_error_h1'].mean()
    worst_hours = by_hour.nlargest(5)
    best_hours = by_hour.nsmallest(5)

    print("  Worst hours:")
    for hour, mae in worst_hours.items():
        print(f"    Hour {hour:2d}: {mae:.1f} MW")
    print("  Best hours:")
    for hour, mae in best_hours.items():
        print(f"    Hour {hour:2d}: {mae:.1f} MW")

    # By day of week
    print("\nMAE by Day of Week:")
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    by_dow = test_df.groupby('dow')['abs_error_h1'].mean()
    for dow, mae in by_dow.items():
        print(f"  {dow_names[dow]}: {mae:.1f} MW")

    # Extreme errors
    print("\nExtreme Error Analysis:")
    large_errors = test_df[test_df['abs_error_h1'] > 80]
    print(f"  Errors > 80 MW: {len(large_errors)} ({len(large_errors)/len(test_df)*100:.1f}%)")

    if len(large_errors) > 0:
        print("  When do they occur?")
        print(f"    Hour distribution: {large_errors['hour'].value_counts().head(5).to_dict()}")
        print(f"    Day distribution: {large_errors['dow'].map(lambda x: dow_names[x]).value_counts().to_dict()}")

    return test_df


def main():
    print("=" * 60)
    print("TRICKY DAYS ANALYSIS")
    print("=" * 60)
    print("\nLooking for hidden patterns in forecast errors...")

    # Load data
    df, load_3min = load_data()

    # Train models and get predictions
    test = train_two_stage_models(df)
    test['date'] = test['datetime'].dt.date

    # Find tricky days
    daily_errors = test.groupby('date')['final_error_h1'].apply(
        lambda x: np.abs(x).mean()
    ).sort_values(ascending=False)

    print("\n" + "=" * 60)
    print("TOP 10 TRICKIEST DAYS")
    print("=" * 60)
    for i, (date, mae) in enumerate(daily_errors.head(10).items(), 1):
        damas_mae = test[test['date'] == date]['error'].abs().mean()
        print(f"  {i:2d}. {date}: Our MAE={mae:.1f} MW (DAMAS: {damas_mae:.1f} MW)")

    # Plot tricky days
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    print("\nTricky days (high error):")
    for i, (date, mae) in enumerate(daily_errors.head(5).items()):
        result = plot_tricky_day(date, test, load_3min, OUTPUT_PATH)
        if result:
            print(f"  Saved: tricky_day_{date}.png (MAE: {mae:.1f} MW)")

    # Also plot normal days for comparison
    print("\nNormal days (median error):")
    median_mae = daily_errors.median()
    normal_days = daily_errors[
        (daily_errors > median_mae * 0.95) &
        (daily_errors < median_mae * 1.05)
    ]
    for i, (date, mae) in enumerate(normal_days.head(2).items()):
        result = plot_tricky_day(date, test, load_3min, OUTPUT_PATH)
        if result:
            print(f"  Saved: tricky_day_{date}.png (MAE: {mae:.1f} MW)")

    # Analyze patterns
    test = analyze_error_patterns(test)

    print(f"\n" + "=" * 60)
    print(f"Plots saved to: {OUTPUT_PATH}")
    print("=" * 60)

    return test, daily_errors


if __name__ == "__main__":
    test, daily_errors = main()
